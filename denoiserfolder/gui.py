import json
import os
import sys
import threading
import queue
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

import torch
import torch.nn.functional as F

from PIL import Image
from PyQt6 import QtCore, QtGui, QtWidgets
from safetensors.torch import load_file as load_safetensors

from denoiser.models import BiasFreeResidualDenoiser

PATCH_SIZE = 256
IMAGE_GLOBS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
QUALITY_GEOM_TRANSFORMS: List[Tuple[int, bool]] = [
    (0, False),
    (1, False),
    (2, False),
    (3, False),
    (0, True),
    (1, True),
    (2, True),
    (3, True),
]
QUALITY_GAIN_FACTORS: Tuple[float, ...] = (0.95, 1.0, 1.05)


def normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(0.0, 1.0)


def denormalize_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(0.0, 1.0)


def load_image_tensor(path: Path, channels: int = 3) -> torch.Tensor:
    mode = "L" if channels == 1 else "RGB"
    with Image.open(path) as img:
        img = img.convert(mode)
        array = np.asarray(img, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[..., None]
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return normalize_tensor(tensor)


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    if t.dim() == 4:
        t = t[0]
    tensor = denormalize_tensor(t).detach().cpu()
    if tensor.shape[0] == 1:
        arr = tensor.squeeze(0).numpy()
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")
    arr = tensor.permute(1, 2, 0).numpy()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)



def pil_to_pixmap(image: Image.Image) -> QtGui.QPixmap:
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    qimage = QtGui.QImage(
        data,
        image.width,
        image.height,
        QtGui.QImage.Format.Format_RGBA8888,
    )
    return QtGui.QPixmap.fromImage(qimage)

def split_into_patches(
    tensor: torch.Tensor,
    patch_size: int,
    overlap: int,
) -> Tuple[torch.Tensor, List[Tuple[int, int]], Tuple[int, int], Tuple[int, int]]:
    stride = patch_size - overlap
    if stride <= 0:
        raise ValueError("Overlap must be smaller than the patch size.")

    _, height, width = tensor.shape
    grid_h = max(1, (height - overlap + stride - 1) // stride)
    grid_w = max(1, (width - overlap + stride - 1) // stride)

    pad_h = max(0, stride * (grid_h - 1) + patch_size - height)
    pad_w = max(0, stride * (grid_w - 1) + patch_size - width)
    padded = F.pad(tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").squeeze(0)
    _, padded_h, padded_w = padded.shape

    positions_h = list(range(0, padded_h - patch_size + 1, stride))
    if positions_h[-1] != padded_h - patch_size:
        positions_h.append(padded_h - patch_size)

    positions_w = list(range(0, padded_w - patch_size + 1, stride))
    if positions_w[-1] != padded_w - patch_size:
        positions_w.append(padded_w - patch_size)

    patches = []
    coords = []
    for top in positions_h:
        for left in positions_w:
            patch = padded[:, top : top + patch_size, left : left + patch_size]
            patches.append(patch)
            coords.append((top, left))

    stacked = torch.stack(patches, 0)
    return stacked, coords, (height, width), (padded_h, padded_w)


def assemble_from_patches(
    patches: torch.Tensor,
    coords: Sequence[Tuple[int, int]],
    original_size: Tuple[int, int],
    padded_size: Tuple[int, int],
    patch_size: int,
    overlap: int,
) -> torch.Tensor:
    padded_h, padded_w = padded_size
    device = patches.device
    channels = patches.shape[1]
    result = torch.zeros(channels, padded_h, padded_w, device=device)
    weights = torch.zeros(1, padded_h, padded_w, device=device)

    stride = patch_size - overlap
    if overlap > 0:
        window = torch.hann_window(patch_size, periodic=False, dtype=torch.float32, device=device)
        weight_patch = torch.outer(window, window).unsqueeze(0)
    else:
        weight_patch = torch.ones(1, patch_size, patch_size, device=device)

    for patch, (top, left) in zip(patches, coords):
        result[:, top : top + patch_size, left : left + patch_size] += patch * weight_patch
        weights[:, top : top + patch_size, left : left + patch_size] += weight_patch

    weights = weights.clamp(min=1e-6)
    result = result / weights
    height, width = original_size
    result = result[:, :height, :width]
    return result


class PreviewGraphicsView(QtWidgets.QGraphicsView):
    scrolled = QtCore.pyqtSignal(int, int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor("#202020")))
        self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._suppress_scroll = False

    def set_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        scene = self.scene()
        scene.clear()
        if not pixmap.isNull():
            item = scene.addPixmap(pixmap)
            scene.setSceneRect(item.boundingRect())
        else:
            scene.setSceneRect(QtCore.QRectF())
        self.resetTransform()

    def set_zoom(self, factor: float) -> None:
        factor = max(0.05, float(factor))
        self.resetTransform()
        self.scale(factor, factor)

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        if not self._suppress_scroll:
            self.scrolled.emit(
                self.horizontalScrollBar().value(),
                self.verticalScrollBar().value(),
            )

    def sync_scroll(self, h_value: int, v_value: int) -> None:
        self._suppress_scroll = True
        self.horizontalScrollBar().setValue(h_value)
        self.verticalScrollBar().setValue(v_value)
        self._suppress_scroll = False


class ComparisonPreview(QtWidgets.QGroupBox):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Single Image Preview", parent)
        self.setMinimumWidth(320)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.stack = QtWidgets.QStackedLayout()
        layout.addLayout(self.stack)

        placeholder = QtWidgets.QWidget()
        placeholder_layout = QtWidgets.QVBoxLayout(placeholder)
        placeholder_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label = QtWidgets.QLabel(
            "Preview activates automatically when a single image is processed."
        )
        self.placeholder_label.setWordWrap(True)
        self.placeholder_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        placeholder_layout.addWidget(self.placeholder_label)
        self.stack.addWidget(placeholder)

        self.preview_container = QtWidgets.QWidget()
        views_layout = QtWidgets.QHBoxLayout(self.preview_container)
        views_layout.setSpacing(8)

        left_column = QtWidgets.QVBoxLayout()
        right_column = QtWidgets.QVBoxLayout()
        for column in (left_column, right_column):
            column.setSpacing(6)

        self.original_label = QtWidgets.QLabel("Original")
        self.original_view = PreviewGraphicsView()
        left_column.addWidget(self.original_label)
        left_column.addWidget(self.original_view, 1)

        self.denoised_label = QtWidgets.QLabel("Denoised")
        self.denoised_view = PreviewGraphicsView()
        right_column.addWidget(self.denoised_label)
        right_column.addWidget(self.denoised_view, 1)

        views_layout.addLayout(left_column, 1)
        views_layout.addLayout(right_column, 1)
        self.stack.addWidget(self.preview_container)

        self.original_view.scrolled.connect(
            lambda h, v: self.denoised_view.sync_scroll(h, v)
        )
        self.denoised_view.scrolled.connect(
            lambda h, v: self.original_view.sync_scroll(h, v)
        )

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(6)
        self.zoom_out_btn = QtWidgets.QToolButton()
        self.zoom_out_btn.setText("−")
        self.zoom_in_btn = QtWidgets.QToolButton()
        self.zoom_in_btn.setText("+")
        self.zoom_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(25, 400)
        self.zoom_slider.setSingleStep(5)
        self.zoom_slider.setValue(100)
        self.zoom_value_label = QtWidgets.QLabel("100%")
        controls.addWidget(self.zoom_out_btn)
        controls.addWidget(self.zoom_slider, 1)
        controls.addWidget(self.zoom_in_btn)
        controls.addWidget(self.zoom_value_label)
        layout.addLayout(controls)
        self.controls_widget = controls

        self.zoom_slider.valueChanged.connect(self._apply_zoom)
        self.zoom_in_btn.clicked.connect(lambda: self._nudge_zoom(10))
        self.zoom_out_btn.clicked.connect(lambda: self._nudge_zoom(-10))

        self.reset()

    def reset(self) -> None:
        self.stack.setCurrentIndex(0)
        self.placeholder_label.setText(
            "Preview activates automatically when a single image is processed."
        )
        self.zoom_slider.setValue(100)
        self._apply_zoom(self.zoom_slider.value())
        self._set_controls_enabled(False)

    def show_waiting(self, source: Path) -> None:
        self.stack.setCurrentIndex(0)
        self.placeholder_label.setText(
            f"Processing:\n{source.name}"
        )
        self._set_controls_enabled(False)

    def show_images(self, original: Path, denoised: Path) -> None:
        original_pix = QtGui.QPixmap(str(original))
        denoised_pix = QtGui.QPixmap(str(denoised))
        if original_pix.isNull() or denoised_pix.isNull():
            self.stack.setCurrentIndex(0)
            self.placeholder_label.setText("Unable to load preview images.")
            self._set_controls_enabled(False)
            return
        self.original_label.setText(f"Original ({original.name})")
        self.denoised_label.setText(f"Denoised ({denoised.name})")
        self.original_view.set_pixmap(original_pix)
        self.denoised_view.set_pixmap(denoised_pix)
        self.stack.setCurrentIndex(1)
        self.zoom_slider.setValue(100)
        self._apply_zoom(self.zoom_slider.value())
        self._set_controls_enabled(True)

    def _nudge_zoom(self, delta: int) -> None:
        value = self.zoom_slider.value()
        self.zoom_slider.setValue(int(np.clip(value + delta, 25, 400)))

    def _apply_zoom(self, value: int) -> None:
        factor = max(0.05, value / 100.0)
        self.original_view.set_zoom(factor)
        self.denoised_view.set_zoom(factor)
        self.zoom_value_label.setText(f"{value}%")

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.zoom_slider.setEnabled(enabled)
        self.zoom_in_btn.setEnabled(enabled)
        self.zoom_out_btn.setEnabled(enabled)
        self.zoom_value_label.setEnabled(enabled)



class RestorationModel:
    def __init__(self):
        self.model: BiasFreeResidualDenoiser | None = None
        self.device = torch.device("cpu")
        self.channels = 3
        self.loaded_checkpoint: Path | None = None
        self.checkpoint_config: Dict[str, Any] | None = None

    def load(self, checkpoint_path: Path, device: torch.device) -> None:
        resolved = checkpoint_path.resolve()
        state_dict, config = self._read_checkpoint(resolved, device)
        model_kwargs = self._infer_model_kwargs(state_dict, config)
        model = BiasFreeResidualDenoiser(**model_kwargs)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        self.model = model
        self.device = device
        self.channels = model_kwargs["image_channels"]
        self.loaded_checkpoint = resolved
        self.checkpoint_config = config

    def ensure_loaded(self, checkpoint_path: Path, device: torch.device) -> None:
        resolved = checkpoint_path.resolve()
        if (
            self.model is None
            or self.device != device
            or self.loaded_checkpoint != resolved
        ):
            self.load(resolved, device)

    def restore(
        self,
        image_path: Path,
        output_path: Path,
        patch_size: int,
        overlap: int,
        batch_size: int,
        repeat_passes: int = 2,
        quality_mode: bool = True,
        tto_steps: int = 0,
        tto_tv_weight: float = 0.0,
        tto_lr: float = 0.0,
        noise_level: float = 25.0,
        auto_noise: bool = True,
    ) -> float:
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        current = load_image_tensor(image_path, channels=self.channels)
        sigma_value = float(noise_level) / 255.0
        if auto_noise:
            estimated = self._estimate_noise_level(current)
            if estimated > 0.0:
                sigma_value = estimated
        sigma_value = float(max(5e-4, min(1.0, sigma_value)))

        repeat_passes = max(1, int(repeat_passes))
        quality_mode = bool(quality_mode)

        for _ in range(repeat_passes):
            patches, coords, original_size, padded_size = split_into_patches(current, patch_size, overlap)
            patches = patches.to(self.device)
            effective_batch = max(1, min(batch_size, patches.size(0)))

            restored_batches = []
            with torch.no_grad():
                for start in range(0, patches.size(0), effective_batch):
                    batch = patches[start : start + effective_batch]
                    if quality_mode:
                        restored = self._quality_tta_pass(batch, sigma_value)
                    else:
                        restored = self._run_model(batch, sigma_value)
                    restored_batches.append(restored)

            restored_all = torch.cat(restored_batches, dim=0)
            assembled = assemble_from_patches(restored_all, coords, original_size, padded_size, patch_size, overlap)
            if tto_steps > 0 and tto_tv_weight > 0.0 and tto_lr > 0.0:
                current = self._apply_test_time_optimization(assembled, tto_steps, tto_tv_weight, tto_lr)
            else:
                current = assembled.detach().cpu()

        image = tensor_to_image(current)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return sigma_value * 255.0

    def _run_model(self, batch: torch.Tensor, sigma_value: float) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        sigma_value = float(max(5e-4, min(1.0, sigma_value)))
        sigma = torch.full(
            (batch.shape[0], 1),
            sigma_value,
            dtype=batch.dtype,
            device=batch.device,
        )
        denoised, _ = self.model(batch, sigma)
        return denoised

    def _quality_tta_pass(self, batch: torch.Tensor, sigma_value: float) -> torch.Tensor:
        accum = torch.zeros_like(batch)
        total_weight = 0.0
        for rotation, flip in QUALITY_GEOM_TRANSFORMS:
            augmented = self._apply_geom_transform(batch, rotation, flip)
            for gain in QUALITY_GAIN_FACTORS:
                scaled = torch.clamp(augmented * gain, 0.0, 1.0)
                pred = self._run_model(scaled, sigma_value)
                if gain != 0.0:
                    pred = torch.clamp(pred / gain, 0.0, 1.0)
                pred = self._invert_geom_transform(pred, rotation, flip)
                accum = accum + pred
                total_weight += 1.0
        if total_weight == 0.0:
            return self._run_model(batch, sigma_value)
        return accum / total_weight


    @staticmethod
    def _apply_geom_transform(tensor: torch.Tensor, rotation: int, hflip: bool) -> torch.Tensor:
        result = tensor
        if rotation % 4:
            result = torch.rot90(result, k=rotation % 4, dims=(-2, -1))
        if hflip:
            result = torch.flip(result, dims=(-1,))
        return result

    @staticmethod
    def _invert_geom_transform(tensor: torch.Tensor, rotation: int, hflip: bool) -> torch.Tensor:
        result = tensor
        if hflip:
            result = torch.flip(result, dims=(-1,))
        inv_rot = (-rotation) % 4
        if inv_rot:
            result = torch.rot90(result, k=inv_rot, dims=(-2, -1))
        return result

    @staticmethod
    def _total_variation(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 4:
            raise ValueError("Expected tensor with shape (B,C,H,W) or (C,H,W) for TV computation.")
        dh = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
        dw = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
        return dh.abs().mean() + dw.abs().mean()

    def _apply_test_time_optimization(
        self,
        tensor: torch.Tensor,
        steps: int,
        tv_weight: float,
        lr: float,
    ) -> torch.Tensor:
        steps = max(0, int(steps))
        tv_weight = max(0.0, float(tv_weight))
        lr = max(0.0, float(lr))
        if steps == 0 or tv_weight == 0.0 or lr == 0.0:
            return tensor.detach().cpu()

        variable = tensor.detach().clone().requires_grad_(True)
        target = tensor.detach()
        optimizer = torch.optim.Adam([variable], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            tv = self._total_variation(variable)
            fidelity = F.mse_loss(variable, target)
            loss = fidelity + tv_weight * tv
            loss.backward()
            optimizer.step()
            variable.data.clamp_(0.0, 1.0)

        return variable.detach().cpu()

    def _read_checkpoint(
        self, checkpoint_path: Path, device: torch.device
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any] | None]:
        suffix = checkpoint_path.suffix.lower()
        if suffix in {".pth", ".pt"}:
            data = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
            if isinstance(data, dict) and "model" in data:
                state_dict = data["model"]
                config = data.get("config")
            else:
                state_dict = data
                config = None
            return state_dict, config
        if suffix == ".safetensors":
            state_dict = load_safetensors(str(checkpoint_path))
            config = self._load_sidecar_config(checkpoint_path)
            return state_dict, config
        raise ValueError(f"Unsupported checkpoint extension '{checkpoint_path.suffix}'.")

    def _load_sidecar_config(self, checkpoint_path: Path) -> Dict[str, Any] | None:
        candidates = [
            checkpoint_path.with_suffix(".json"),
            checkpoint_path.with_name(f"{checkpoint_path.stem}_config.json"),
            checkpoint_path.with_name("config.json"),
            checkpoint_path.with_name("training_config.json"),
            checkpoint_path.parent / "config.json",
            checkpoint_path.parent / "training_config.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                try:
                    return json.loads(candidate.read_text())
                except json.JSONDecodeError:
                    continue
        summary = checkpoint_path.parent / "model_summary.txt"
        if summary.exists():
            text = summary.read_text()
            start = text.find("{")
            end = text.rfind("}")
            if 0 <= start < end:
                snippet = text[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    pass
        return None

    def _infer_model_kwargs(
        self, state_dict: Dict[str, torch.Tensor], config: Dict[str, Any] | None
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if config:
            color_mode = str(config.get("color_mode", "rgb")).lower()
            kwargs["image_channels"] = 1 if color_mode.startswith("gray") else 3
            kernels = config.get("residual_kernel_sizes")
            if kernels:
                kwargs["residual_kernel_sizes"] = tuple(int(k) for k in kernels)
        exit_weight = state_dict.get("exit.weight")
        if exit_weight is not None:
            kwargs["image_channels"] = exit_weight.shape[0]
        entry_weight = state_dict.get("entry.weight")
        if entry_weight is not None:
            kwargs["base_channels"] = entry_weight.shape[0]
        depth = self._count_indices(state_dict, "encoder_stages.")
        if depth:
            kwargs["depth"] = depth
        blocks = self._count_indices(state_dict, "encoder_stages.0.blocks.")
        if blocks:
            kwargs["blocks_per_stage"] = blocks
        if "residual_kernel_sizes" not in kwargs:
            kwargs["residual_kernel_sizes"] = self._infer_kernel_sizes(state_dict)
        kwargs.setdefault("image_channels", 3)
        kwargs.setdefault("base_channels", 48)
        kwargs.setdefault("depth", 3)
        kwargs.setdefault("blocks_per_stage", 2)
        kwargs.setdefault("residual_kernel_sizes", (3, 5, 7))
        return kwargs

    @staticmethod
    def _count_indices(state_dict: Dict[str, torch.Tensor], prefix: str) -> int:
        indices: set[int] = set()
        prefix_len = len(prefix)
        for key in state_dict.keys():
            if key.startswith(prefix):
                remainder = key[prefix_len:]
                parts = remainder.split(".")
                if parts and parts[0].isdigit():
                    indices.add(int(parts[0]))
        return len(indices)

    @staticmethod
    def _infer_kernel_sizes(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, ...]:
        kernels: List[int] = []
        for key, tensor in state_dict.items():
            if key.startswith("prior_extractor.kernel_"):
                size = int(tensor.shape[-1])
                kernels.append(size)
        kernels.sort()
        return tuple(kernels) if kernels else (3, 5, 7)

    @staticmethod
    def _estimate_noise_level(tensor: torch.Tensor) -> float:
        data = tensor.detach().cpu()
        if data.dim() == 4:
            data = data[0]
        if data.shape[0] > 1:
            gray = 0.2989 * data[0] + 0.5870 * data[1] + 0.1140 * data[2]
        else:
            gray = data[0]
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3)
        response = F.conv2d(gray.unsqueeze(0).unsqueeze(0), kernel, padding=1)
        sigma = response.abs().median() / 0.6745
        value = float(sigma.item())
        if not np.isfinite(value):
            return 0.0
        return float(max(0.0, min(1.0, value)))


class SettingsManager:
    """Manages application settings and remembered paths."""
    
    def __init__(self) -> None:
        self.config_dir = Path.home() / ".config" / "batch_denoiser"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "settings.json"
        self.settings = self._load_settings()
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from JSON file."""
        if self.config_file.exists():
            try:
                return json.loads(self.config_file.read_text())
            except (json.JSONDecodeError, IOError):
                return self._default_settings()
        return self._default_settings()
    
    def _default_settings(self) -> Dict[str, Any]:
        """Return default settings."""
        return {
            "last_checkpoint_dir": str(Path.home()),
            "last_image_dir": str(Path.home()),
            "last_folder_dir": str(Path.home()),
            "last_output_dir": str(Path.home()),
        }
    
    def save_settings(self) -> None:
        """Save settings to JSON file."""
        try:
            self.config_file.write_text(json.dumps(self.settings, indent=2))
        except IOError:
            pass
    
    def get_last_checkpoint_dir(self) -> str:
        """Get last checkpoint directory."""
        path_str = self.settings.get("last_checkpoint_dir", str(Path.home()))
        path = Path(path_str)
        return str(path) if path.exists() else str(Path.home())
    
    def set_last_checkpoint_dir(self, path: Path) -> None:
        """Set last checkpoint directory."""
        self.settings["last_checkpoint_dir"] = str(path.parent)
        self.save_settings()
    
    def get_last_image_dir(self) -> str:
        """Get last image selection directory."""
        path_str = self.settings.get("last_image_dir", str(Path.home()))
        path = Path(path_str)
        return str(path) if path.exists() else str(Path.home())
    
    def set_last_image_dir(self, path: Path) -> None:
        """Set last image selection directory."""
        self.settings["last_image_dir"] = str(path.parent)
        self.save_settings()
    
    def get_last_folder_dir(self) -> str:
        """Get last folder selection directory."""
        path_str = self.settings.get("last_folder_dir", str(Path.home()))
        path = Path(path_str)
        return str(path) if path.exists() else str(Path.home())
    
    def set_last_folder_dir(self, path: Path) -> None:
        """Set last folder selection directory."""
        self.settings["last_folder_dir"] = str(path)
        self.save_settings()
    
    def get_last_output_dir(self) -> str:
        """Get last output directory."""
        path_str = self.settings.get("last_output_dir", str(Path.home()))
        path = Path(path_str)
        return str(path) if path.exists() else str(Path.home())
    
    def set_last_output_dir(self, path: Path) -> None:
        """Set last output directory."""
        self.settings["last_output_dir"] = str(path)
        self.save_settings()


def iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        return [path]
    images = []
    for pattern in IMAGE_GLOBS:
        images.extend(path.rglob(pattern))
    unique = {img.resolve(strict=False): img for img in images}
    return [unique[key] for key in sorted(unique, key=lambda p: str(p).lower())]


class BatchDenoiseWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Batch Image Denoiser")
        self.resize(780, 560)
        self._apply_dark_theme()

        self.queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.restorer = RestorationModel()
        self.settings_manager = SettingsManager()

        self._input_paths: List[str] = []
        self._input_keys: set[str] = set()
        self.preview_enabled = False
        self._preview_source: Path | None = None

        self._build_ui()

        self._queue_timer = QtCore.QTimer(self)
        self._queue_timer.timeout.connect(self._poll_queue)
        self._queue_timer.start(150)

    def _apply_dark_theme(self) -> None:
        base = QtGui.QColor("#111111")
        alt = QtGui.QColor("#1a1a1a")
        text = QtGui.QColor("#f3f3f3")
        highlight = QtGui.QColor("#ff7f32")
        palette = self.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, base)
        palette.setColor(QtGui.QPalette.ColorRole.Base, alt)
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#202020"))
        palette.setColor(QtGui.QPalette.ColorRole.Text, text)
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#1f1f1f"))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, text)
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, highlight)
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#101010"))
        self.setPalette(palette)
        self.setStyleSheet(
            """
            QWidget {
                background-color: #0f0f0f;
                color: #f3f3f3;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #292929;
                border-radius: 6px;
                margin-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
                color: #ff8a3d;
                font-weight: 600;
            }
            QLineEdit,
            QPlainTextEdit,
            QTextEdit,
            QListWidget,
            QSpinBox,
            QDoubleSpinBox,
            QComboBox {
                background-color: #1d1d1d;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 3px;
                selection-background-color: #ff7f32;
                selection-color: #111111;
            }
            QLineEdit:focus,
            QPlainTextEdit:focus,
            QTextEdit:focus,
            QListWidget:focus,
            QSpinBox:focus,
            QDoubleSpinBox:focus,
            QComboBox:focus {
                border: 1px solid #ff7f32;
            }
            QListWidget::item:selected {
                background-color: #ff7f32;
                color: #111111;
            }
            QPushButton,
            QToolButton {
                background-color: #1f1f1f;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 4px 10px;
            }
            QPushButton:hover,
            QToolButton:hover {
                border: 1px solid #ff7f32;
                color: #ffffff;
            }
            QPushButton:disabled,
            QToolButton:disabled {
                color: #777777;
                border-color: #2a2a2a;
            }
            QCheckBox::indicator {
                border: 1px solid #3a3a3a;
                width: 16px;
                height: 16px;
                background: #1d1d1d;
            }
            QCheckBox::indicator:checked {
                background-color: #ff7f32;
                border: 1px solid #ff7f32;
            }
            QComboBox::drop-down {
                border: 0px;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0px;
            }
            QProgressBar {
                background-color: #1d1d1d;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                text-align: center;
                color: #f3f3f3;
            }
            QProgressBar::chunk {
                background-color: #ff7f32;
                border-radius: 4px;
            }
            QScrollBar:vertical,
            QScrollBar:horizontal {
                background: #1a1a1a;
                border: none;
                margin: 0px;
            }
            QScrollBar::handle {
                background: #3f3f3f;
                border-radius: 4px;
            }
            QScrollBar::handle:hover {
                background: #ff7f32;
            }
            QSlider::groove:horizontal {
                border: 1px solid #3a3a3a;
                height: 6px;
                background: #1f1f1f;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ff7f32;
                border: 1px solid #ff7f32;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QTabWidget::pane {
                border: 1px solid #292929;
                background: #0f0f0f;
            }
            QTabBar::tab {
                background: #111111;
                color: #ff7f32;
                padding: 6px 12px;
                border: 1px solid #292929;
                border-bottom: none;
                min-width: 60px;
            }
            QTabBar::tab:selected {
                background: #1f1f1f;
            }
            QTabBar::tab:!selected {
                margin-top: 4px;
            }
            """
        )

    def _build_ui(self) -> None:
        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        self.main_tabs = QtWidgets.QTabWidget()
        self.main_tabs.setDocumentMode(True)
        root_layout.addWidget(self.main_tabs)

        workflow_widget = QtWidgets.QWidget()
        workflow_layout = QtWidgets.QHBoxLayout(workflow_widget)
        workflow_layout.setContentsMargins(4, 4, 4, 4)
        workflow_layout.setSpacing(14)

        left_container = QtWidgets.QWidget()
        left_container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        left_container.setMaximumWidth(652)
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setSpacing(14)
        workflow_layout.addWidget(
            left_container,
            0,
            QtCore.Qt.AlignmentFlag.AlignLeft,
        )

        model_group = QtWidgets.QGroupBox("Model")
        model_layout = QtWidgets.QGridLayout()
        model_layout.setHorizontalSpacing(12)
        model_layout.setVerticalSpacing(8)
        model_layout.setColumnStretch(1, 1)
        model_group.setLayout(model_layout)

        self.checkpoint_edit = QtWidgets.QLineEdit()
        self.checkpoint_edit.setPlaceholderText("Select a .pt or .safetensors checkpoint file")
        checkpoint_button = QtWidgets.QPushButton("Browse…")
        self.check_button = checkpoint_button
        checkpoint_button.clicked.connect(self._select_checkpoint)

        model_layout.addWidget(QtWidgets.QLabel("Checkpoint (.pt/.safetensors):"), 0, 0)
        model_layout.addWidget(self.checkpoint_edit, 0, 1)
        model_layout.addWidget(checkpoint_button, 0, 2)

        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItem("Auto", "auto")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("CUDA", "cuda")
        model_layout.addWidget(QtWidgets.QLabel("Device:"), 1, 0)
        model_layout.addWidget(self.device_combo, 1, 1)

        self.noise_spin = QtWidgets.QDoubleSpinBox()
        self.noise_spin.setDecimals(1)
        self.noise_spin.setRange(0.0, 100.0)
        self.noise_spin.setSingleStep(0.5)
        self.noise_spin.setValue(25.0)
        self.noise_spin.setSuffix(" σ")
        self.noise_spin.setToolTip("Assumed noise standard deviation expressed on [0, 100].")

        self.auto_noise_checkbox = QtWidgets.QCheckBox("Auto-estimate per image")
        self.auto_noise_checkbox.setChecked(True)
        self.auto_noise_checkbox.toggled.connect(self._on_auto_noise_toggled)
        self._on_auto_noise_toggled(self.auto_noise_checkbox.isChecked())

        model_layout.addWidget(QtWidgets.QLabel("Noise level:"), 2, 0)
        model_layout.addWidget(self.noise_spin, 2, 1)
        model_layout.addWidget(self.auto_noise_checkbox, 2, 2)

        left_layout.addWidget(model_group)

        inputs_group = QtWidgets.QGroupBox("Inputs")
        inputs_layout = QtWidgets.QVBoxLayout()
        inputs_layout.setSpacing(8)
        inputs_group.setLayout(inputs_layout)

        controls_row = QtWidgets.QHBoxLayout()
        controls_row.setSpacing(8)

        self.add_images_btn = QtWidgets.QPushButton("Add Images…")
        self.add_images_btn.clicked.connect(self._add_images)
        controls_row.addWidget(self.add_images_btn)

        self.add_folder_btn = QtWidgets.QPushButton("Add Folder…")
        self.add_folder_btn.clicked.connect(self._add_folder)
        controls_row.addWidget(self.add_folder_btn)

        self.remove_selected_btn = QtWidgets.QPushButton("Remove Selected")
        self.remove_selected_btn.clicked.connect(self._remove_selected)
        controls_row.addWidget(self.remove_selected_btn)

        self.clear_inputs_btn = QtWidgets.QPushButton("Clear All")
        self.clear_inputs_btn.clicked.connect(self._clear_inputs)
        controls_row.addWidget(self.clear_inputs_btn)

        controls_row.addStretch(1)
        inputs_layout.addLayout(controls_row)

        self.input_list = QtWidgets.QListWidget()
        self.input_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.input_list.setAlternatingRowColors(True)
        self.input_list.setMinimumHeight(180)
        inputs_layout.addWidget(self.input_list)

        self.input_summary = QtWidgets.QLabel("No items added.")
        self.input_summary.setStyleSheet("color: #666666;")
        inputs_layout.addWidget(self.input_summary)

        output_row = QtWidgets.QHBoxLayout()
        output_row.setSpacing(8)
        output_row.addWidget(QtWidgets.QLabel("Output folder:"))

        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setPlaceholderText("Leave blank to create a 'denoised' folder beside your inputs")
        output_row.addWidget(self.output_edit, 1)

        output_button = QtWidgets.QPushButton("Browse…")
        output_button.clicked.connect(self._select_output)
        output_row.addWidget(output_button)

        inputs_layout.addLayout(output_row)

        left_layout.addWidget(inputs_group)

        options_group = QtWidgets.QGroupBox("Tiling")
        options_layout = QtWidgets.QGridLayout()
        options_layout.setHorizontalSpacing(12)
        options_layout.setVerticalSpacing(8)
        options_group.setLayout(options_layout)

        self.patch_spin = QtWidgets.QSpinBox()
        self.patch_spin.setRange(64, 4096)
        self.patch_spin.setSingleStep(32)
        self.patch_spin.setValue(256)

        self.overlap_spin = QtWidgets.QSpinBox()
        self.overlap_spin.setRange(0, 2048)
        self.overlap_spin.setSingleStep(8)
        self.overlap_spin.setValue(48)

        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(8)

        self.repeat_spin = QtWidgets.QSpinBox()
        self.repeat_spin.setRange(1, 10)
        self.repeat_spin.setValue(2)
        self.repeat_spin.setToolTip("Number of times each tile is reprocessed before blending.")

        self.quality_checkbox = QtWidgets.QCheckBox("Maximum quality TTA (slow)")
        self.quality_checkbox.setChecked(True)
        self.quality_checkbox.setToolTip("Runs a heavy set of test-time augmentations for best quality (much slower).")

        options_layout.addWidget(QtWidgets.QLabel("Patch size:"), 0, 0)
        options_layout.addWidget(self.patch_spin, 0, 1)
        options_layout.addWidget(QtWidgets.QLabel("Overlap:"), 1, 0)
        options_layout.addWidget(self.overlap_spin, 1, 1)
        options_layout.addWidget(QtWidgets.QLabel("Tiles per batch:"), 2, 0)
        options_layout.addWidget(self.batch_spin, 2, 1)
        options_layout.addWidget(QtWidgets.QLabel("Repeats per tile:"), 3, 0)
        options_layout.addWidget(self.repeat_spin, 3, 1)
        options_layout.addWidget(self.quality_checkbox, 4, 0, 1, 2)

        left_layout.addWidget(options_group)

        refinement_group = QtWidgets.QGroupBox("Refinement")
        refinement_layout = QtWidgets.QGridLayout()
        refinement_layout.setHorizontalSpacing(12)
        refinement_layout.setVerticalSpacing(8)
        refinement_group.setLayout(refinement_layout)

        self.tto_checkbox = QtWidgets.QCheckBox("Test-time optimization (slow)")
        self.tto_checkbox.setToolTip("Run a short optimization on each pass to enforce simple priors (adds latency).")
        self.tto_checkbox.toggled.connect(self._on_tto_toggled)

        self.tto_steps_spin = QtWidgets.QSpinBox()
        self.tto_steps_spin.setRange(1, 50)
        self.tto_steps_spin.setValue(5)
        self.tto_steps_spin.setToolTip("Gradient steps per optimization pass.")

        self.tto_tv_spin = QtWidgets.QDoubleSpinBox()
        self.tto_tv_spin.setDecimals(4)
        self.tto_tv_spin.setRange(0.0000, 0.1000)
        self.tto_tv_spin.setSingleStep(0.0005)
        self.tto_tv_spin.setValue(0.0050)
        self.tto_tv_spin.setToolTip("Weight for total-variation smoothing prior.")

        self.tto_lr_spin = QtWidgets.QDoubleSpinBox()
        self.tto_lr_spin.setDecimals(3)
        self.tto_lr_spin.setRange(0.001, 0.5)
        self.tto_lr_spin.setSingleStep(0.01)
        self.tto_lr_spin.setValue(0.05)
        self.tto_lr_spin.setToolTip("Step size for the optimization updates.")

        refinement_layout.addWidget(self.tto_checkbox, 0, 0, 1, 2)
        refinement_layout.addWidget(QtWidgets.QLabel("Steps:"), 1, 0)
        refinement_layout.addWidget(self.tto_steps_spin, 1, 1)
        refinement_layout.addWidget(QtWidgets.QLabel("TV weight:"), 2, 0)
        refinement_layout.addWidget(self.tto_tv_spin, 2, 1)
        refinement_layout.addWidget(QtWidgets.QLabel("Learning rate:"), 3, 0)
        refinement_layout.addWidget(self.tto_lr_spin, 3, 1)

        self._on_tto_toggled(self.tto_checkbox.isChecked())

        left_layout.addWidget(refinement_group)

        progress_layout = QtWidgets.QVBoxLayout()
        progress_layout.setSpacing(8)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QtWidgets.QLabel("Idle.")
        self.status_label.setMinimumHeight(24)
        progress_layout.addWidget(self.status_label)

        action_row = QtWidgets.QHBoxLayout()
        action_row.addStretch(1)
        self.start_button = QtWidgets.QPushButton("Start Denoising")
        self.start_button.clicked.connect(self._start)
        action_row.addWidget(self.start_button)

        progress_layout.addLayout(action_row)
        left_layout.addLayout(progress_layout)

        self.preview_panel = ComparisonPreview()
        self.preview_panel.setVisible(False)
        workflow_layout.addWidget(self.preview_panel, 1)
        self.main_tabs.addTab(workflow_widget, "Workflow")

    @staticmethod
    def _normalize_path(path: Path) -> Path:
        expanded = path.expanduser()
        try:
            return expanded.resolve(strict=False)
        except Exception:
            return expanded

    @staticmethod
    def _path_key(path: Path) -> str:
        normalized = BatchDenoiseWindow._normalize_path(path)
        return os.path.normcase(str(normalized))

    def _select_checkpoint(self) -> None:
        initial_dir = self.settings_manager.get_last_checkpoint_dir()
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select checkpoint",
            initial_dir,
            "Checkpoints (*.pt *.safetensors);;PyTorch (*.pt);;Safetensors (*.safetensors);;All files (*.*)",
        )
        if filename:
            self.checkpoint_edit.setText(filename)
            self.settings_manager.set_last_checkpoint_dir(Path(filename))

    def _select_output(self) -> None:
        initial_dir = self.settings_manager.get_last_output_dir()
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            initial_dir,
        )
        if directory:
            self.output_edit.setText(directory)
            self.settings_manager.set_last_output_dir(Path(directory))

    def _add_images(self) -> None:
        initial_dir = self.settings_manager.get_last_image_dir()
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select images",
            initial_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
        )
        added = False
        for file_path in files:
            path_obj = Path(file_path)
            key = self._path_key(path_obj)
            if key in self._input_keys:
                continue
            self._input_keys.add(key)
            self._input_paths.append(str(self._normalize_path(path_obj)))
            added = True
            if added:
                self.settings_manager.set_last_image_dir(path_obj)
        if added:
            self._update_input_list()

    def _add_folder(self) -> None:
        initial_dir = self.settings_manager.get_last_folder_dir()
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select folder",
            initial_dir,
        )
        if not directory:
            return
        path_obj = Path(directory)
        key = self._path_key(path_obj)
        if key in self._input_keys:
            return
        self._input_keys.add(key)
        self._input_paths.append(str(self._normalize_path(path_obj)))
        self.settings_manager.set_last_folder_dir(path_obj)
        self._update_input_list()

    def _remove_selected(self) -> None:
        selected_items = self.input_list.selectedItems()
        if not selected_items:
            return
        keys_to_remove = {os.path.normcase(item.data(QtCore.Qt.ItemDataRole.UserRole)) for item in selected_items}
        if not keys_to_remove:
            return
        self._input_paths = [p for p in self._input_paths if os.path.normcase(p) not in keys_to_remove]
        self._input_keys.difference_update(keys_to_remove)
        self._update_input_list()

    def _clear_inputs(self) -> None:
        if not self._input_paths:
            return
        self._input_paths.clear()
        self._input_keys.clear()
        self._update_input_list()

    def _update_input_list(self) -> None:
        self.input_list.clear()
        for path_str in self._input_paths:
            item = QtWidgets.QListWidgetItem(path_str)
            item.setToolTip(path_str)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path_str)
            self.input_list.addItem(item)
        count = len(self._input_paths)
        if count == 0:
            self.input_summary.setText("No items added.")
        else:
            plural = "s" if count != 1 else ""
            self.input_summary.setText(f"{count} item{plural} queued.")

    def _collect_tasks(self) -> List[Tuple[Path, Path]]:
        tasks: List[Tuple[Path, Path]] = []
        seen: set[str] = set()
        for path_str in self._input_paths:
            entry = Path(path_str)
            if not entry.exists():
                raise FileNotFoundError(f"Input not found: {path_str}")
            if entry.is_dir():
                for image_path in iter_images(entry):
                    key = self._path_key(image_path)
                    if key in seen:
                        continue
                    seen.add(key)
                    relative = image_path.relative_to(entry)
                    tasks.append((image_path, relative))
            else:
                if entry.suffix.lower() not in IMAGE_EXTENSIONS:
                    raise ValueError(f"{entry.name} is not a supported image file.")
                key = self._path_key(entry)
                if key in seen:
                    continue
                seen.add(key)
                tasks.append((entry, Path(entry.name)))
        return tasks

    def _default_output_root(self) -> Path:
        if not self._input_paths:
            return Path.cwd() / "denoised"
        first_entry = Path(self._input_paths[0])
        base = first_entry.parent if first_entry.is_file() else first_entry
        return base / "denoised"

    def _on_tto_toggled(self, checked: bool) -> None:
        widgets = (self.tto_steps_spin, self.tto_tv_spin, self.tto_lr_spin)
        for widget in widgets:
            widget.setEnabled(checked)

    def _on_auto_noise_toggled(self, checked: bool) -> None:
        self.noise_spin.setEnabled(not checked)

    def _configure_preview(self, enabled: bool, source: Path | None) -> None:
        self.preview_enabled = enabled
        self._preview_source = source
        if not enabled:
            self.preview_panel.reset()
            self.preview_panel.setVisible(False)
            return
        self.preview_panel.setVisible(True)
        if source is not None:
            self.preview_panel.show_waiting(source)
        else:
            self.preview_panel.reset()

    def _resolve_device_choice(self) -> torch.device:
        device_choice = self.device_combo.currentData(
            QtCore.Qt.ItemDataRole.UserRole
        )
        if device_choice is None:
            device_choice = self.device_combo.currentText().lower()
        if device_choice == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_choice == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available on this machine.")
            return torch.device("cuda")
        return torch.device("cpu")

    def _start(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            QtWidgets.QMessageBox.information(self, "Processing", "Processing is already running.")
            return

        checkpoint_text = self.checkpoint_edit.text().strip()
        if not checkpoint_text:
            QtWidgets.QMessageBox.warning(self, "Checkpoint", "Please choose a checkpoint file.")
            return
        checkpoint = self._normalize_path(Path(checkpoint_text))
        if not checkpoint.exists():
            QtWidgets.QMessageBox.critical(self, "Checkpoint", "Selected checkpoint file does not exist.")
            return

        if not self._input_paths:
            QtWidgets.QMessageBox.warning(self, "Inputs", "Please add at least one image or folder.")
            return

        try:
            tasks = self._collect_tasks()
        except FileNotFoundError as exc:
            QtWidgets.QMessageBox.critical(self, "Missing input", str(exc))
            return
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Unsupported input", str(exc))
            return

        if not tasks:
            QtWidgets.QMessageBox.information(self, "No images", "No images found in the selected inputs.")
            return

        output_text = self.output_edit.text().strip()
        if output_text:
            output_root = self._normalize_path(Path(output_text))
        else:
            output_root = self._default_output_root()
        output_root.mkdir(parents=True, exist_ok=True)

        patch_size = int(self.patch_spin.value())
        overlap = int(self.overlap_spin.value())
        batch_size = int(self.batch_spin.value())
        repeat_passes = int(self.repeat_spin.value())
        quality_mode = self.quality_checkbox.isChecked()
        tto_enabled = self.tto_checkbox.isChecked()
        tto_steps = int(self.tto_steps_spin.value()) if tto_enabled else 0
        tto_tv_weight = float(self.tto_tv_spin.value()) if tto_enabled else 0.0
        tto_lr = float(self.tto_lr_spin.value()) if tto_enabled else 0.0
        noise_level = float(self.noise_spin.value())
        auto_noise = self.auto_noise_checkbox.isChecked()

        if overlap < 0 or overlap >= patch_size:
            QtWidgets.QMessageBox.warning(self, "Overlap", "Overlap must be zero or less than the patch size.")
            return

        try:
            device = self._resolve_device_choice()
        except RuntimeError as exc:
            QtWidgets.QMessageBox.critical(self, "Device", str(exc))
            return
        preview_single = len(tasks) == 1
        preview_source = tasks[0][0] if preview_single else None
        self._configure_preview(preview_single, preview_source)

        self.progress_bar.setValue(0)
        self.status_label.setText("Preparing restoration model…")
        self._set_running(True)

        self.worker_thread = threading.Thread(
            target=self._worker,
            args=(
                checkpoint,
                tasks,
                output_root,
                patch_size,
                overlap,
                batch_size,
                repeat_passes,
                quality_mode,
                tto_steps,
                tto_tv_weight,
                tto_lr,
                noise_level,
                auto_noise,
                preview_single,
                device,
            ),
            daemon=True,
        )
        self.worker_thread.start()

    def _worker(
        self,
        checkpoint: Path,
        tasks: List[Tuple[Path, Path]],
        output_root: Path,
        patch_size: int,
        overlap: int,
        batch_size: int,
        repeat_passes: int,
        quality_mode: bool,
        tto_steps: int,
        tto_tv_weight: float,
        tto_lr: float,
        noise_level: float,
        auto_noise: bool,
        preview_enabled: bool,
        device: torch.device,
    ) -> None:
        try:
            self.restorer.ensure_loaded(checkpoint, device)
            total = len(tasks)
            for idx, (image_path, relative) in enumerate(tasks, start=1):
                output_path = output_root / relative
                output_path.parent.mkdir(parents=True, exist_ok=True)
                used_sigma = self.restorer.restore(
                    image_path,
                    output_path,
                    patch_size,
                    overlap,
                    batch_size,
                    repeat_passes,
                    quality_mode,
                    tto_steps,
                    tto_tv_weight,
                    tto_lr,
                    noise_level,
                    auto_noise,
                )
                if preview_enabled:
                    self.queue.put(("preview", str(image_path), str(output_path)))
                self.queue.put(
                    (
                        "progress",
                        idx,
                        total,
                        f"Denoised {idx}/{total} • σ={used_sigma:.1f}",
                    )
                )
            self.queue.put(("done", str(output_root)))
        except Exception as exc:
            self.queue.put(("error", str(exc)))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, *payload = self.queue.get_nowait()
                if kind == "progress":
                    done, total, message = payload
                    pct = int(round(100.0 * done / max(total, 1)))
                    self.progress_bar.setValue(pct)
                    self.status_label.setText(message)
                elif kind == "preview":
                    source, output = payload
                    if self.preview_enabled and self.preview_panel.isVisible():
                        self.preview_panel.show_images(Path(source), Path(output))
                elif kind == "done":
                    folder = payload[0]
                    self.progress_bar.setValue(100)
                    self.status_label.setText("All images processed.")
                    QtWidgets.QMessageBox.information(self, "Completed", f"Denoised images saved under:\n{folder}")
                    self._set_running(False)
                elif kind == "error":
                    message = payload[0]
                    self.status_label.setText("Error encountered.")
                    QtWidgets.QMessageBox.critical(self, "Error", message)
                    self._set_running(False)
        except queue.Empty:
            pass

    def _set_running(self, running: bool) -> None:
        controls = [
            self.checkpoint_edit,
            self.device_combo,
            self.add_images_btn,
            self.add_folder_btn,
            self.remove_selected_btn,
            self.clear_inputs_btn,
            self.input_list,
            self.output_edit,
            self.patch_spin,
            self.overlap_spin,
            self.batch_spin,
            self.repeat_spin,
            self.quality_checkbox,
            self.noise_spin,
            self.auto_noise_checkbox,
            self.tto_checkbox,
            self.tto_steps_spin,
            self.tto_tv_spin,
            self.tto_lr_spin,
            self.check_button,
        ]
        for widget in controls:
            widget.setEnabled(not running)
        self.start_button.setText("Processing…" if running else "Start Denoising")
        self.start_button.setEnabled(not running)
        if not running:
            self.worker_thread = None
            self._on_tto_toggled(self.tto_checkbox.isChecked())
            self._on_auto_noise_toggled(self.auto_noise_checkbox.isChecked())


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = BatchDenoiseWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


