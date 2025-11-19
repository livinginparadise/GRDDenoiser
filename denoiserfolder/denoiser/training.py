from __future__ import annotations

import json
import math
import pickle
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, Tuple
import pathlib

import torch
import torch.nn.functional as F
from torch.cuda import amp
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from safetensors.torch import save_file as save_safetensors

try:  # Torch < 2.6 does not expose add_safe_globals.
    from torch.serialization import add_safe_globals
except (ImportError, AttributeError):
    add_safe_globals = None

from .config import TrainingConfig
from .data import build_dataloaders
from .models import BiasFreeResidualDenoiser
from .utils import (
    AverageMeter,
    ExponentialMovingAverage,
    count_parameters,
    plot_metrics,
    psnr,
    set_seed,
    visualize_epoch_summary,
)


class CharbonnierLoss(torch.nn.Module):
    def __init__(self, epsilon: float = 1e-3, reduction: str = "mean"):
        super().__init__()
        self.epsilon = epsilon
        if reduction not in {"mean", "none"}:
            raise ValueError("Reduction must be 'mean' or 'none'.")
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        if self.reduction == "mean":
            return loss.mean()
        return loss.flatten(1).mean(dim=1)


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        self.padding = window_size // 2
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu_x = F.avg_pool2d(pred, self.window_size, stride=1, padding=self.padding)
        mu_y = F.avg_pool2d(target, self.window_size, stride=1, padding=self.padding)
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x = (
            F.avg_pool2d(pred * pred, self.window_size, 1, self.padding) - mu_x_sq
        )
        sigma_y = (
            F.avg_pool2d(target * target, self.window_size, 1, self.padding) - mu_y_sq
        )
        sigma_xy = (
            F.avg_pool2d(pred * target, self.window_size, 1, self.padding) - mu_xy
        )

        ssim_map = ((2 * mu_xy + self.c1) * (2 * sigma_xy + self.c2)) / (
            (mu_x_sq + mu_y_sq + self.c1) * (sigma_x + sigma_y + self.c2)
        )
        ssim_map = torch.clamp(ssim_map, 0.0, 1.0)
        ssim_per_sample = 1.0 - ssim_map.flatten(1).mean(dim=1)
        return ssim_per_sample


class GradientLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer("kernel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        grad_pred_x, grad_pred_y = self._compute_grads(pred)
        grad_target_x, grad_target_y = self._compute_grads(target)
        grad_diff = torch.abs(grad_pred_x - grad_target_x) + torch.abs(
            grad_pred_y - grad_target_y
        )
        return grad_diff.flatten(1).mean(dim=1)

    def _compute_grads(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c = tensor.shape[1]
        kernel_x = self.kernel_x.to(tensor.device).repeat(c, 1, 1, 1)
        kernel_y = self.kernel_y.to(tensor.device).repeat(c, 1, 1, 1)
        grad_x = F.conv2d(tensor, kernel_x, padding=1, groups=c)
        grad_y = F.conv2d(tensor, kernel_y, padding=1, groups=c)
        return grad_x, grad_y


class SpectralLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_residual: torch.Tensor, target_residual: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred_residual, norm="ortho")
        target_fft = torch.fft.rfft2(target_residual, norm="ortho")
        return torch.mean(torch.abs(pred_fft - target_fft))


class HardExampleBuffer:
    def __init__(self, capacity: int, batch_size: int):
        self.capacity = capacity
        self.batch_size = batch_size
        self.storage: deque = deque(maxlen=capacity)

    def add(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        losses: torch.Tensor,
        topk: int,
    ) -> None:
        if losses.numel() == 0:
            return
        k = min(topk, losses.numel())
        _, indices = torch.topk(losses.detach(), k=k)
        for idx in indices.tolist():
            self.storage.append(
                {
                    "clean": clean[idx].detach().cpu(),
                    "noisy": noisy[idx].detach().cpu(),
                    "sigma": sigma[idx].detach().cpu(),
                }
            )

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if len(self.storage) < self.batch_size:
            return None
        indices = random.sample(range(len(self.storage)), self.batch_size)
        clean = torch.stack([self.storage[i]["clean"] for i in indices], dim=0)
        noisy = torch.stack([self.storage[i]["noisy"] for i in indices], dim=0)
        sigma = torch.stack([self.storage[i]["sigma"] for i in indices], dim=0)
        return clean, noisy, sigma


class BlindDenoiserTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.cfg = config
        set_seed(config.seed)
        config.ensure_dirs()

        self.train_loader, self.eval_loader = build_dataloaders(config)

        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model = BiasFreeResidualDenoiser(
            image_channels=config.channels,
            base_channels=48,
            depth=3,
            blocks_per_stage=2,
            residual_kernel_sizes=config.residual_kernel_sizes,
        ).to(self.device)
        self.param_info = count_parameters(self.model)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.total_steps = (
            len(self.train_loader) * config.num_epochs
        ) // max(1, config.grad_accum)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_lambda
        )

        self.autocast = config.precision == "amp" and torch.cuda.is_available()
        self.scaler = amp.GradScaler(enabled=self.autocast)
        self.global_step = 0
        self.start_epoch = 1
        self.ema = ExponentialMovingAverage(self.model, decay=config.ema_decay)
        self.charbonnier = CharbonnierLoss(reduction="none")
        self.ssim_loss = SSIMLoss()
        self.gradient_loss = GradientLoss()
        self.spectral_loss = SpectralLoss()
        self.hard_buffer = (
            HardExampleBuffer(self.cfg.hard_buffer_size, self.cfg.batch_size)
            if self.cfg.hard_mining
            else None
        )
        self._path_globals_registered = False

        if config.resume_checkpoint and config.resume_checkpoint.exists():
            self._load_checkpoint(config.resume_checkpoint)

        self.train_sample_dir = self.cfg.output_dir / "samples" / "train"
        self.eval_sample_dir = self.cfg.output_dir / "samples" / "eval"
        self._write_model_summary()

    def _write_model_summary(self) -> None:
        params = getattr(self, "param_info", count_parameters(self.model))
        total = int(params["total"])
        trainable = int(params["trainable"])
        summary_path = self.cfg.output_dir / "model_summary.txt"
        cfg_serializable = self._serialized_config()
        lines = [
            "Model Summary",
            "=============",
            f"Total parameters: {total:,} ({total / 1e6:.3f} M)",
            f"Trainable parameters: {trainable:,} ({trainable / 1e6:.3f} M)",
            "",
            "Training Config:",
            json.dumps(cfg_serializable, indent=2),
            "",
            "Architecture:",
            str(self.model),
        ]
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        print(
            f"[Model] Parameters: {total / 1e6:.2f}M "
            f"(trainable {trainable / 1e6:.2f}M). Summary -> {summary_path}"
        )

    def _serialized_config(self) -> Dict[str, Any]:
        return {
            key: self._serialize_config_value(value)
            for key, value in self.cfg.__dict__.items()
        }

    def _serialize_config_value(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: self._serialize_config_value(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return [self._serialize_config_value(v) for v in value]
        if isinstance(value, list):
            return [self._serialize_config_value(v) for v in value]
        return value

    def _lr_lambda(self, step: int) -> float:
        if self.total_steps == 0:
            return 1.0
        warmup = self.cfg.warmup_steps
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        progress = (step - warmup) / float(max(1, self.total_steps - warmup))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def fit(self) -> None:
        history: Dict[str, list] = {"train_loss": [], "eval_psnr": []}
        best_psnr = 0.0
        best_loss = float("inf")

        for epoch in range(self.start_epoch, self.cfg.num_epochs + 1):
            train_loss = self._train_epoch(epoch)
            history["train_loss"].append(train_loss)
            best_loss = min(best_loss, train_loss)

            should_eval = epoch % self.cfg.sample_interval == 0 or epoch == self.cfg.num_epochs
            if should_eval:
                metrics = self.evaluate(epoch)
                history["eval_psnr"].append(metrics["psnr"])
                best_psnr = max(best_psnr, metrics["psnr"])

            self._save_checkpoint(epoch, best_psnr, best_loss)

        plot_metrics(history, self.cfg.output_dir / "training_metrics.png")
        with open(self.cfg.output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        loss_meter = AverageMeter("loss")
        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.cfg.num_epochs}",
            leave=False,
        )
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress, start=1):
            clean, noisy, sigma = self._prepare_batch(batch)
            with amp.autocast(enabled=self.autocast):
                (
                    loss,
                    per_sample,
                    denoised_batch,
                    residual_batch,
                ) = self._forward_loss(clean, noisy, sigma)

            scaled_loss = loss / self.cfg.grad_accum
            self.scaler.scale(scaled_loss).backward()

            if step % self.cfg.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.ema.update(self.model)
                self.global_step += 1

            loss_meter.update(loss.item(), clean.size(0))
            self._maybe_log_train_sample(
                epoch,
                step,
                clean,
                noisy,
                sigma,
                denoised_batch,
                residual_batch,
            )
            if self.cfg.hard_mining and self.hard_buffer is not None:
                self.hard_buffer.add(
                    clean, noisy, sigma, per_sample.detach(), self.cfg.hard_mining_topk
                )
                if (
                    self.global_step > 0
                    and self.global_step % self.cfg.hard_mining_interval == 0
                ):
                    self._optimize_on_hard_examples()
            if step % self.cfg.log_interval == 0:
                progress.set_postfix(
                    loss=f"{loss_meter.avg:.4f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )

        return loss_meter.avg

    def _forward_loss(
        self, clean: torch.Tensor, noisy: torch.Tensor, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        denoised, residual = self.model(noisy, sigma)
        residual_target = noisy - clean
        recon_loss = self.charbonnier(denoised, clean)
        residual_loss = self.charbonnier(residual, residual_target)
        ssim_loss = self.ssim_loss(denoised, clean)
        grad_loss = self.gradient_loss(denoised, clean)
        spectral_loss = self.spectral_loss(residual, residual_target)
        total = (
            recon_loss
            + 0.5 * residual_loss
            + self.cfg.ssim_weight * ssim_loss
            + self.cfg.gradient_weight * grad_loss
            + self.cfg.spectral_weight * spectral_loss
        )
        return total.mean(), total, denoised, residual

    def _maybe_log_train_sample(
        self,
        epoch: int,
        step: int,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        denoised: torch.Tensor,
        residual_pred: torch.Tensor,
    ) -> None:
        interval = self.cfg.sample_step_interval
        if interval <= 0 or step % interval != 0:
            return
        self.train_sample_dir.mkdir(parents=True, exist_ok=True)
        clean_sample = clean[:1].detach()
        noisy_sample = noisy[:1].detach()
        denoised_sample = denoised[:1].detach()
        residual_pred_sample = residual_pred[:1].detach()
        residual_target_sample = (noisy_sample - clean_sample).detach()
        sigma_val = float(sigma[:1].mean().item())
        meta = {
            "Epoch": float(epoch),
            "Train step": float(step),
            "Noise σ": sigma_val,
            "PSNR": psnr(denoised_sample, clean_sample),
        }
        output_path = (
            self.train_sample_dir / f"epoch_{epoch:04d}_step_{step:06d}.png"
        )
        visualize_epoch_summary(
            clean_sample,
            noisy_sample,
            denoised_sample,
            residual_pred_sample,
            residual_target_sample,
            output_path,
            meta=meta,
        )

    def _optimize_on_hard_examples(self) -> None:
        if not self.hard_buffer:
            return
        sample = self.hard_buffer.sample()
        if sample is None:
            return
        clean, noisy, sigma = sample
        clean = clean.to(self.device, non_blocking=True)
        noisy = noisy.to(self.device, non_blocking=True)
        sigma = sigma.to(self.device, non_blocking=True)
        with amp.autocast(enabled=self.autocast):
            loss, _, _, _ = self._forward_loss(clean, noisy, sigma)
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.ema.update(self.model)
        self.global_step += 1

    def _prepare_batch(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clean = batch["clean"].to(self.device, non_blocking=True)
        noisy = batch["noisy"].to(self.device, non_blocking=True)
        sigma = batch["sigma"].to(self.device, non_blocking=True)
        return clean, noisy, sigma

    @torch.no_grad()
    def evaluate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        self.ema.apply_shadow(self.model)
        meter = AverageMeter("psnr")
        samples_logged = 0
        max_samples = max(0, self.cfg.visualize_samples_per_epoch)
        samples_dir = self.eval_sample_dir
        if max_samples > 0:
            samples_dir.mkdir(parents=True, exist_ok=True)
        for batch in self.eval_loader:
            clean, noisy, sigma = self._prepare_batch(batch)
            denoised, residual_pred = self.model(noisy, sigma)
            meter.update(psnr(denoised, clean), n=1)
            if samples_logged < max_samples:
                level = batch["noise_level"]
                if isinstance(level, torch.Tensor):
                    level_value = float(level[0].item())
                elif isinstance(level, (list, tuple)):
                    level_value = float(level[0])
                else:
                    level_value = float(level)
                residual_target = noisy - clean
                meta = {
                    "Epoch": float(epoch),
                    "Noise σ": level_value / 255.0,
                    "PSNR": psnr(denoised, clean),
                }
                visualize_epoch_summary(
                    clean[:1],
                    noisy[:1],
                    denoised[:1],
                    residual_pred[:1],
                    residual_target[:1],
                    samples_dir / f"epoch_{epoch:04d}_sample_{samples_logged:02d}.png",
                    meta=meta,
                )
                samples_logged += 1
        self.ema.restore(self.model)
        return {"psnr": meter.avg}

    def _save_checkpoint(self, epoch: int, best_psnr: float, best_loss: float) -> None:
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ema": self.ema.shadow,
            "config": self._serialized_config(),
            "best_psnr": best_psnr,
            "best_loss": best_loss,
        }
        path_pt = self.cfg.output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        path_sft = self.cfg.output_dir / f"checkpoint_epoch_{epoch:04d}.safetensors"
        torch.save(ckpt, path_pt)
        save_safetensors(
            self.model.state_dict(),
            str(path_sft),
            metadata={
                "epoch": str(epoch),
                "best_psnr": f"{best_psnr:.4f}",
                "best_loss": f"{best_loss:.6f}",
            },
        )

    def _load_checkpoint(self, path: Path) -> None:
        ckpt = self._safe_load_checkpoint(path)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.ema.shadow = ckpt.get("ema", self.ema.shadow)
        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["epoch"] * len(self.train_loader)

    def _safe_load_checkpoint(self, path: Path) -> Dict[str, Any]:
        load_kwargs = {"map_location": self.device}
        try:
            return torch.load(path, **load_kwargs)
        except pickle.UnpicklingError:
            registered = self._register_path_safe_globals()
            if registered:
                try:
                    return torch.load(path, **load_kwargs)
                except pickle.UnpicklingError:
                    pass
            print(
                f"[Checkpoint] Unpickling failed for {path}. "
                "Retrying with weights_only=False."
            )
            return self._torch_load_allow_code(path, load_kwargs)

    def _torch_load_allow_code(
        self, path: Path, load_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        kwargs = dict(load_kwargs)
        try:
            kwargs["weights_only"] = False
            return torch.load(path, **kwargs)
        except TypeError:
            kwargs.pop("weights_only", None)
            return torch.load(path, **kwargs)

    def _register_path_safe_globals(self) -> bool:
        if self._path_globals_registered or add_safe_globals is None:
            return self._path_globals_registered
        safe_types = set()
        for attr in (
            "Path",
            "PurePath",
            "PosixPath",
            "PurePosixPath",
            "WindowsPath",
            "PureWindowsPath",
        ):
            cls = getattr(pathlib, attr, None)
            if isinstance(cls, type):
                safe_types.add(cls)
        safe_types.add(type(Path()))
        if not safe_types:
            return False
        add_safe_globals(list(safe_types))
        self._path_globals_registered = True
        return True
