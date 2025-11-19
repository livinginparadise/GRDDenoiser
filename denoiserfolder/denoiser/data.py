from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import TrainingConfig

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list_images(root: Path) -> List[Path]:
    files = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in _IMAGE_EXTS and p.is_file()
    )
    if not files:
        raise FileNotFoundError(
            f"No image files with extensions {_IMAGE_EXTS} found under {root}"
        )
    return files


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    return tensor


def _resize_min_side(img: Image.Image, min_side: int) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    if side >= min_side:
        return img
    scale = min_side / float(side)
    new_size = int(round(w * scale)), int(round(h * scale))
    resample = getattr(Image, "Resampling", Image).BICUBIC
    return img.resize(new_size, resample=resample)


def _random_crop(tensor: torch.Tensor, size: int) -> torch.Tensor:
    _, h, w = tensor.shape
    if h == size and w == size:
        return tensor
    if h < size or w < size:
        raise ValueError(
            f"Requested crop {size} but tensor spatial dims {(h, w)} are smaller."
        )
    top = random.randint(0, h - size)
    left = random.randint(0, w - size)
    return tensor[:, top : top + size, left : left + size]


def _center_crop(tensor: torch.Tensor, size: int) -> torch.Tensor:
    _, h, w = tensor.shape
    if h < size or w < size:
        raise ValueError(
            f"Requested crop {size} but tensor spatial dims {(h, w)} are smaller."
        )
    top = (h - size) // 2
    left = (w - size) // 2
    return tensor[:, top : top + size, left : left + size]


def _augment(tensor: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        tensor = tensor.flip(-1)
    if random.random() < 0.5:
        tensor = tensor.flip(-2)
    if random.random() < 0.5:
        tensor = tensor.transpose(-1, -2)
    return tensor


def _match_paired_files(clean_root: Path, noisy_root: Path) -> List[Tuple[Path, Path]]:
    clean_files = _list_images(clean_root)
    clean_map = {p.name: p for p in clean_files}
    pairs: List[Tuple[Path, Path]] = []
    for name, c_path in clean_map.items():
        n_path = noisy_root / name
        if n_path.exists():
            pairs.append((c_path, n_path))
    if not pairs:
        raise FileNotFoundError(
            f"No paired files found between {clean_root} and {noisy_root}."
        )
    return sorted(pairs, key=lambda x: x[0].name)


def _apply_joint_transform(
    clean: torch.Tensor, noisy: torch.Tensor, transform_fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    stacked = torch.cat([clean, noisy], dim=0)
    stacked = transform_fn(stacked)
    c = clean.shape[0]
    return stacked[:c], stacked[c:]


class BlindNoisyDataset(Dataset):
    """
    Dataset generating random noisy crops for blind denoising.

    Each clean image is randomly augmented, cropped, and corrupted by
    Gaussian noise sampled from a schedule of training noise levels.
    """

    def __init__(
        self,
        root: Path,
        patch_size: int,
        noise_levels: Sequence[float],
        color_mode: str,
        train: bool = True,
        max_images: int | None = None,
        paired: bool = False,
        clean_subdir: str = "clean",
        noisy_subdir: str = "noisy",
        residual_aug_prob: float = 0.0,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.train = train
        self.paired = paired
        self.clean_subdir = clean_subdir
        self.noisy_subdir = noisy_subdir
        self.residual_aug_prob = max(0.0, residual_aug_prob)
        self.noise_levels = tuple(float(s) for s in noise_levels)
        if not self.noise_levels and not paired:
            raise ValueError("At least one noise level must be provided.")
        self.patch_size = patch_size
        mode = "L" if color_mode.lower().startswith("gray") else "RGB"
        self.color_mode = mode
        self.max_images = max_images

        if self.paired:
            clean_root = self.root / self.clean_subdir
            noisy_root = self.root / self.noisy_subdir
            pairs = _match_paired_files(clean_root, noisy_root)
            if max_images:
                pairs = pairs[:max_images]
            self.pairs = pairs
        else:
            paths = _list_images(self.root)
            if max_images:
                paths = paths[:max_images]
            self.paths = paths

        if not train and not self.paired:
            self.eval_index: List[Tuple[Path, float]] = [
                (path, level) for path in self.paths for level in self.noise_levels
            ]

    def __len__(self) -> int:
        if self.paired:
            return len(self.pairs)
        if self.train:
            return len(self.paths)
        return len(self.eval_index)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert(self.color_mode)
        img = _resize_min_side(img, self.patch_size)
        tensor = _pil_to_tensor(img)
        return tensor

    def _sample_noise_level(self, index: int) -> float:
        if self.paired:
            return 0.0
        if self.train:
            base = self.noise_levels
            # Encourage interpolation between provided levels.
            if len(base) > 1:
                low, high = sorted(random.sample(base, 2))
                alpha = random.random()
                return (1 - alpha) * low + alpha * high
            return random.choice(base)
        return self.eval_index[index][1]

    def _paired_sample(self, index: int) -> dict:
        clean_path, noisy_path = self.pairs[index % len(self.pairs)]
        clean = self._load_image(clean_path)
        noisy = self._load_image(noisy_path)
        if self.train:
            transform = lambda x: _random_crop(x, self.patch_size)
            clean, noisy = _apply_joint_transform(clean, noisy, transform)
            clean, noisy = _apply_joint_transform(clean, noisy, _augment)
        else:
            transform = lambda x: _center_crop(x, self.patch_size)
            clean, noisy = _apply_joint_transform(clean, noisy, transform)
        noise = noisy - clean
        if self.train and self.residual_aug_prob > 0.0:
            if random.random() < self.residual_aug_prob:
                noise = self._residual_variant(noise)
                noisy = torch.clamp(clean + noise, 0.0, 1.0)
        sigma = torch.clamp(noise.reshape(-1).std(), min=1e-4)
        sigma_value = float(sigma.item())
        sigma_level = sigma_value * 255.0
        return {
            "clean": clean,
            "noisy": noisy,
            "sigma": torch.tensor([sigma_value], dtype=torch.float32),
            "noise_level": sigma_level,
            "path": str(clean_path),
        }

    def _residual_variant(self, residual: torch.Tensor) -> torch.Tensor:
        variant = residual
        k = random.randint(0, 3)
        if k:
            variant = torch.rot90(variant, k, dims=(-2, -1))
        if random.random() < 0.5:
            variant = variant.flip(-1)
        if random.random() < 0.5:
            variant = variant.flip(-2)
        return variant

    def __getitem__(self, index: int) -> dict:
        if self.paired:
            return self._paired_sample(index)
        if self.train:
            path = self.paths[index % len(self.paths)]
        else:
            path = self.eval_index[index][0]
        clean = self._load_image(path)
        if self.train:
            clean = _random_crop(clean, self.patch_size)
            clean = _augment(clean)
        else:
            clean = _center_crop(clean, self.patch_size)
        sigma_s = self._sample_noise_level(index)
        sigma = sigma_s / 255.0
        noise = torch.randn_like(clean) * sigma
        noisy = torch.clamp(clean + noise, 0.0, 1.0)
        return {
            "clean": clean,
            "noisy": noisy,
            "sigma": torch.tensor([sigma], dtype=torch.float32),
            "noise_level": float(sigma_s),
            "path": str(path),
        }


def build_dataloaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds = BlindNoisyDataset(
        root=config.dataset_path,
        patch_size=config.patch_size,
        noise_levels=config.train_noise_levels,
        color_mode=config.color_mode,
        train=True,
        max_images=config.max_train_images,
        paired=config.paired,
        clean_subdir=config.clean_subdir,
        noisy_subdir=config.noisy_subdir,
        residual_aug_prob=config.residual_aug_prob,
    )
    eval_ds = BlindNoisyDataset(
        root=config.eval_path,
        patch_size=config.patch_size,
        noise_levels=config.eval_noise_levels,
        color_mode=config.color_mode,
        train=False,
        max_images=None,
        paired=config.paired,
        clean_subdir=config.clean_subdir,
        noisy_subdir=config.noisy_subdir,
        residual_aug_prob=0.0,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )
    return train_loader, eval_loader
