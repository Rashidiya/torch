from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image


# -----------------------------
# Reproducibility and utilities
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Map images from [-1, 1] to [0, 1]."""
    return (x.clamp(-1, 1) + 1.0) / 2.0


def save_grid(images: torch.Tensor, path: Path, nrow: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(images), str(path), nrow=nrow, padding=2)


# -----------------------------
# DDPM schedule and equations
# -----------------------------


@dataclass
class DDPMSchedule:
    T: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    posterior_variance: torch.Tensor

    def to(self, device: torch.device) -> "DDPMSchedule":
        return DDPMSchedule(
            T=self.T,
            betas=self.betas.to(device),
            alphas=self.alphas.to(device),
            alpha_bars=self.alpha_bars.to(device),
            sqrt_alpha_bars=self.sqrt_alpha_bars.to(device),
            sqrt_one_minus_alpha_bars=self.sqrt_one_minus_alpha_bars.to(device),
            sqrt_recip_alphas=self.sqrt_recip_alphas.to(device),
            posterior_variance=self.posterior_variance.to(device),
        )


def make_linear_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> DDPMSchedule:
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)

    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    posterior_variance[0] = 0.0

    return DDPMSchedule(
        T=T,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
        sqrt_recip_alphas=torch.sqrt(1.0 / alphas),
        posterior_variance=posterior_variance,
    )


def extract(values: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Gather values[t] and reshape for broadcasting over image tensors."""
    batch_size = t.shape[0]
    out = values.gather(0, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def q_sample(x0: torch.Tensor, t: torch.Tensor, schedule: DDPMSchedule, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Forward noising: x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon."""
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ab = extract(schedule.sqrt_alpha_bars, t, x0.shape)
    sqrt_omab = extract(schedule.sqrt_one_minus_alpha_bars, t, x0.shape)
    return sqrt_ab * x0 + sqrt_omab * noise


@torch.no_grad()
def p_sample(model: nn.Module, xt: torch.Tensor, t: torch.Tensor, schedule: DDPMSchedule) -> torch.Tensor:
    """One reverse DDPM sampling step from x_t to x_{t-1}."""
    betas_t = extract(schedule.betas, t, xt.shape)
    sqrt_one_minus_ab_t = extract(schedule.sqrt_one_minus_alpha_bars, t, xt.shape)
    sqrt_recip_alpha_t = extract(schedule.sqrt_recip_alphas, t, xt.shape)

    eps_theta = model(xt, t)
    model_mean = sqrt_recip_alpha_t * (xt - betas_t * eps_theta / sqrt_one_minus_ab_t)

    posterior_var_t = extract(schedule.posterior_variance, t, xt.shape)
    noise = torch.randn_like(xt)
    nonzero_mask = (t != 0).float().reshape(xt.shape[0], *((1,) * (len(xt.shape) - 1)))
    return model_mean + nonzero_mask * torch.sqrt(posterior_var_t.clamp(min=1e-20)) * noise


@torch.no_grad()
def sample_images(
    model: nn.Module,
    schedule: DDPMSchedule,
    shape: tuple[int, int, int, int],
    device: torch.device,
    trajectory_steps: Optional[Iterable[int]] = None,
) -> tuple[torch.Tensor, List[torch.Tensor]]:
    """Generate images from pure noise and optionally store a denoising trajectory."""
    model.eval()
    x = torch.randn(shape, device=device)
    trajectory: List[torch.Tensor] = []
    wanted_steps = set(trajectory_steps or [])

    for time_step in reversed(range(schedule.T)):
        t = torch.full((shape[0],), time_step, device=device, dtype=torch.long)
        x = p_sample(model, x, t, schedule)
        if time_step in wanted_steps:
            trajectory.append(x.detach().cpu())

    return x.detach().cpu(), trajectory


# -----------------------------
# Small time-conditioned U-Net
# -----------------------------


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -scale)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ResidualBlock(in_ch, out_ch, time_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x), t_emb)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.block = ResidualBlock(in_ch, out_ch, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.block(x, t_emb)


class SmallUNet(nn.Module):
    def __init__(self, image_channels: int = 1, base_channels: int = 32, time_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.inc = ResidualBlock(image_channels, base_channels, time_dim)
        self.down1 = Down(base_channels, base_channels * 2, time_dim)
        self.down2 = Down(base_channels * 2, base_channels * 4, time_dim)
        self.mid = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        self.up1 = Up(base_channels * 4 + base_channels * 2, base_channels * 2, time_dim)
        self.up2 = Up(base_channels * 2 + base_channels, base_channels, time_dim)
        self.out = nn.Conv2d(base_channels, image_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x1 = self.inc(x, t_emb)      # 28 x 28
        x2 = self.down1(x1, t_emb)   # 14 x 14
        x3 = self.down2(x2, t_emb)   # 7 x 7
        x3 = self.mid(x3, t_emb)
        x = self.up1(x3, x2, t_emb)  # 14 x 14
        x = self.up2(x, x1, t_emb)   # 28 x 28
        return self.out(x)


# -----------------------------
# Training and reporting
# -----------------------------


def get_mnist_loaders(data_dir: Path, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [0, 1] -> [-1, 1]
    ])
    train_set = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def train(
    model: nn.Module,
    schedule: DDPMSchedule,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    max_train_batches: Optional[int] = None,
) -> List[float]:
    model.train()
    losses: List[float] = []

    for epoch in range(1, epochs + 1):
        running = 0.0
        count = 0
        for batch_idx, (x0, _) in enumerate(train_loader, start=1):
            if max_train_batches is not None and batch_idx > max_train_batches:
                break
            x0 = x0.to(device)
            batch_size = x0.shape[0]
            t = torch.randint(0, schedule.T, (batch_size,), device=device).long()
            noise = torch.randn_like(x0)
            xt = q_sample(x0, t, schedule, noise)

            pred_noise = model(xt, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_value = loss.item()
            losses.append(loss_value)
            running += loss_value
            count += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch:03d}/{epochs:03d} | batch {batch_idx:04d} | loss {running / count:.4f}")

        print(f"Epoch {epoch:03d}/{epochs:03d} finished | mean loss {running / max(count, 1):.4f}")

    return losses


@torch.no_grad()
def make_forward_noise_grid(
    schedule: DDPMSchedule,
    test_loader: DataLoader,
    device: torch.device,
    out_path: Path,
    num_images: int = 8,
    num_steps: int = 6,
) -> None:
    x0, _ = next(iter(test_loader))
    x0 = x0[:num_images].to(device)
    timesteps = torch.linspace(0, schedule.T - 1, num_steps, device=device).long()

    rows = []
    for img_idx in range(num_images):
        base = x0[img_idx : img_idx + 1]
        fixed_noise = torch.randn_like(base)
        for t_scalar in timesteps:
            t = t_scalar.repeat(1)
            xt = q_sample(base, t, schedule, fixed_noise)
            rows.append(xt.cpu())
    grid_tensor = torch.cat(rows, dim=0)
    save_grid(grid_tensor, out_path, nrow=num_steps)


def plot_loss_curve(losses: List[float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.xlabel("Training step")
    plt.ylabel("MSE loss")
    plt.title("DDPM noise prediction loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def write_report(
    out_path: Path,
    args: argparse.Namespace,
    losses: List[float],
    image_paths: dict[str, Path],
) -> None:
    final_loss = losses[-1] if losses else float("nan")
    best_loss = min(losses) if losses else float("nan")
    report = f"""# MNIST DDPM Experiment Report

## Objective
Train a small time-conditioned U-Net/CNN DDPM on MNIST 28x28 grayscale digits. Images are normalized to `[-1, 1]`. The model learns to predict the Gaussian noise `epsilon` added at each diffusion timestep.

## Method
The forward process uses a linear beta schedule with `T={args.T}`, `beta_start={args.beta_start}`, and `beta_end={args.beta_end}`:

`x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon`

The reverse model predicts `epsilon_theta(x_t, t)` and is trained with MSE loss:

`L = || epsilon - epsilon_theta(x_t, t) ||^2`

## Hyperparameters
- Epochs: `{args.epochs}`
- Batch size: `{args.batch_size}`
- Learning rate: `{args.lr}`
- Base channels: `{args.base_channels}`
- Device requested: `{args.device}`
- Seed: `{args.seed}`

## Results
- Final training loss: `{final_loss:.6f}`
- Best training loss: `{best_loss:.6f}`

Generated files:
- Forward noising grid: `{image_paths['forward'].name}`
- Reverse denoising trajectory: `{image_paths['reverse'].name}`
- Generated digit grid: `{image_paths['generated'].name}`
- Loss curve: `{image_paths['loss'].name}`
- Model checkpoint: `{image_paths['checkpoint'].name}`

## Notes
Longer training generally improves digit quality. On a GPU, try `--epochs 10 --T 200 --base-channels 32`. For faster debugging on CPU, use `--epochs 1 --T 100 --max-train-batches 20`.
"""
    out_path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST DDPM experiment with forward/reverse grids and loss curve.")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--out-dir", type=Path, default=Path("./ddpm_outputs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--T", type=int, default=200, help="Diffusion timesteps. Use 100 or 200 for this assignment.")
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional cap for quick debugging.")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of generated samples to save.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = default_device()
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = get_mnist_loaders(args.data_dir, args.batch_size, args.num_workers)

    schedule = make_linear_schedule(args.T, args.beta_start, args.beta_end).to(device)
    model = SmallUNet(image_channels=1, base_channels=args.base_channels, time_dim=args.time_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("Saving forward noising grid before training...")
    forward_path = args.out_dir / "forward_noise_grid.png"
    make_forward_noise_grid(schedule, test_loader, device, forward_path)

    print("Training model...")
    losses = train(
        model=model,
        schedule=schedule,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
    )

    checkpoint_path = args.out_dir / "mnist_ddpm_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "losses": losses,
        },
        checkpoint_path,
    )

    print("Sampling generated images and reverse denoising trajectory...")
    num_samples = args.num_samples
    trajectory_steps = torch.linspace(args.T - 1, 0, 8).long().tolist()
    generated, trajectory = sample_images(
        model=model,
        schedule=schedule,
        shape=(num_samples, 1, 28, 28),
        device=device,
        trajectory_steps=trajectory_steps,
    )

    generated_path = args.out_dir / "generated_digits_grid.png"
    save_grid(generated, generated_path, nrow=8)

    # Use the first 8 samples across the saved trajectory. The trajectory is ordered from noisy to clean.
    trajectory_tensor = torch.cat([x[:8] for x in trajectory], dim=0)
    reverse_path = args.out_dir / "reverse_denoising_trajectory.png"
    save_grid(trajectory_tensor, reverse_path, nrow=8)

    loss_path = args.out_dir / "loss_curve.png"
    plot_loss_curve(losses, loss_path)

    report_path = args.out_dir / "mnist_ddpm_report.md"
    write_report(
        report_path,
        args,
        losses,
        {
            "forward": forward_path,
            "reverse": reverse_path,
            "generated": generated_path,
            "loss": loss_path,
            "checkpoint": checkpoint_path,
        },
    )

    print("Done. Files saved to:")
    for p in [forward_path, reverse_path, generated_path, loss_path, checkpoint_path, report_path]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
