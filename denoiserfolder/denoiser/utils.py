from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 99.0
    return 20.0 * np.log10(1.0 / np.sqrt(mse))


def _to_image(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().clamp(0, 1)
    img = tensor.permute(1, 2, 0).numpy()
    if img.shape[2] == 1:
        img = img[..., 0]
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32, copy=False)
    return img


def visualize_triplet(clean, noisy, denoised, output_path: Path, title: str = "") -> None:
    clean = clean.detach().cpu().clamp(0, 1)
    noisy = noisy.detach().cpu().clamp(0, 1)
    denoised = denoised.detach().cpu().clamp(0, 1)
    tensors = [clean, noisy, denoised]
    names = ["Clean", "Noisy", "Denoised"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    if title:
        fig.suptitle(title)
    for ax, tensor, name in zip(axes, tensors, names):
        img = tensor[0] if tensor.dim() == 4 else tensor
        img = img.permute(1, 2, 0).numpy()
        if img.shape[2] == 1:
            img = img[..., 0]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def visualize_epoch_summary(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    denoised: torch.Tensor,
    residual_pred: torch.Tensor,
    residual_target: torch.Tensor,
    output_path: Path,
    meta: Dict[str, float] | None = None,
) -> None:
    clean_i = _to_image(clean)
    noisy_i = _to_image(noisy)
    denoised_i = _to_image(denoised)
    error_map = np.abs(denoised_i - clean_i).astype(np.float32, copy=False)
    res_pred = _to_image(residual_pred * 0.5 + 0.5)
    res_target = _to_image(residual_target * 0.5 + 0.5)
    res_diff = np.abs(res_pred - res_target).astype(np.float32, copy=False)

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1.2])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[1, 3]),
    ]
    titles = [
        "Clean",
        "Noisy",
        "Denoised",
        "Abs Error",
        "Pred Residual",
        "Target Residual",
        "Residual Diff",
        "Residual Histogram",
    ]
    images = [
        clean_i,
        noisy_i,
        denoised_i,
        error_map,
        res_pred,
        res_target,
        res_diff,
    ]
    cmaps = [None, None, None, "magma", "coolwarm", "coolwarm", "magma"]
    for ax, title, img, cmap in zip(axes[:-1], titles[:-1], images, cmaps):
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap or "gray", vmin=0, vmax=1)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    hist_ax = axes[-1]
    pred_flat = residual_pred.detach().cpu().flatten().numpy()
    target_flat = residual_target.detach().cpu().flatten().numpy()
    bins = np.linspace(-0.5, 0.5, 60)
    hist_ax.hist(
        pred_flat,
        bins=bins,
        alpha=0.6,
        label="Pred",
        color="#ff7f0e",
        density=True,
    )
    hist_ax.hist(
        target_flat,
        bins=bins,
        alpha=0.5,
        label="Target",
        color="#1f77b4",
        density=True,
    )
    hist_ax.set_title("Residual Distribution")
    hist_ax.legend()
    hist_ax.grid(True, linestyle="--", alpha=0.3)
    meta = meta or {}
    text_lines = [f"{k}: {v:.3f}" for k, v in meta.items()]
    hist_ax.text(
        0.02,
        0.95,
        "\n".join(text_lines),
        transform=hist_ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_metrics(history: Dict[str, List[float]], output_path: Path) -> None:
    if not history:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, values in history.items():
        ax.plot(values, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


@dataclass
class AverageMeter:
    name: str
    value: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.value += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.value / max(1, self.count)


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            new_avg = (1.0 - self.decay) * param.detach() + self.decay * self.shadow[name]
            self.shadow[name] = new_avg

    def apply_shadow(self, model: torch.nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def count_parameters(model: torch.nn.Module) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": float(total), "trainable": float(trainable)}
