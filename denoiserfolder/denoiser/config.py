from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence, Tuple


def _tuple_floats(values: Iterable[float]) -> Tuple[float, ...]:
    return tuple(float(v) for v in values)


@dataclass
class TrainingConfig:
    """
    Configuration describing the blind denoiser training run.

    The defaults follow the bias-free residual denoising findings from
    "Robust and Interpretable Blind Image Denoising via Bias-free CNNs"
    (Mohan et al., ICLR 2020) by emphasizing residual prediction and
    noise-level generalization.
    """

    dataset_path: Path = Path("data/train")
    eval_path: Path = Path("data/val")
    output_dir: Path = Path("runs/bias_free_denoiser")

    patch_size: int = 256
    paired: bool = False
    clean_subdir: str = "clean"
    noisy_subdir: str = "noisy"
    train_noise_levels: Tuple[float, ...] = field(
        default_factory=lambda: _tuple_floats((5.0, 15.0, 25.0, 50.0))
    )
    eval_noise_levels: Tuple[float, ...] = field(
        default_factory=lambda: _tuple_floats((5.0, 15.0, 25.0, 50.0, 75.0))
    )
    color_mode: str = "rgb"  # or "grayscale"

    batch_size: int = 8
    grad_accum: int = 1
    num_epochs: int = 200
    learning_rate: float = 2e-4
    weight_decay: float = 1e-6
    ema_decay: float = 0.999
    warmup_steps: int = 200
    ssim_weight: float = 0.15
    gradient_weight: float = 0.05
    spectral_weight: float = 0.1
    residual_kernel_sizes: Tuple[int, ...] = field(
        default_factory=lambda: (3, 5, 7)
    )
    residual_aug_prob: float = 0.35
    visualize_samples_per_epoch: int = 2

    num_workers: int = 4
    device: str = "cuda"
    precision: str = "amp"  # amp, fp32, bf16

    log_interval: int = 25
    sample_interval: int = 1
    sample_step_interval: int = 500
    checkpoint_interval: int = 5

    max_train_images: int | None = None
    pin_memory: bool = True
    seed: int = 17
    resume_checkpoint: Path | None = None
    hard_mining: bool = True
    hard_buffer_size: int = 256
    hard_mining_topk: int = 2
    hard_mining_interval: int = 50

    @property
    def channels(self) -> int:
        return 1 if self.color_mode.lower().startswith("gray") else 3

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
