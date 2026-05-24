import json
import os
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn.functional as F

from extensions.alpha_lora_trainer.AlphaLoraTrainer import AlphaLoraTrainer
from toolkit.print import print_acc


class AlphaLoraEdgeLockTrainer(AlphaLoraTrainer):
    """
    Alpha-conditioned LoRA trainer with:

      1) weighted flow matching via dataset mask_path -> batch.mask_tensor
      2) edge-locked structure loss anchored to the input control image

    This keeps the old dataloader/training route intact: the experiment still
    uses the standard alpha_lora_trainer batching path rather than the
    same-sample pair loader added for structural-consistency ablations.
    """

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        edge_conf = self.get_conf("edge_lock", {}) or {}

        self.edge_lock_enabled: bool = bool(edge_conf.get("enabled", True))
        self.edge_loss_weight: float = float(edge_conf.get("lambda_edge", 0.05))
        self.edge_warmup_ratio: float = float(edge_conf.get("warmup_ratio", 0.30))
        self.edge_every: int = max(1, int(edge_conf.get("every", 4)))
        self.edge_min_t: float = float(edge_conf.get("min_t", 0.5))
        self.edge_sample_fraction: float = float(
            edge_conf.get("sample_fraction", 0.5)
        )
        self.edge_max_samples: int = int(edge_conf.get("max_samples", 1))
        self.edge_grad_threshold: float = float(
            edge_conf.get("grad_threshold", 0.05)
        )
        self.edge_log_eps: float = float(edge_conf.get("log_eps", 1e-6))
        self.edge_cos_eps: float = float(edge_conf.get("cos_eps", 1e-6))
        self.edge_normalize_mode: str = str(
            edge_conf.get("normalize_mode", "none")
        ).lower()
        self.edge_stage2_lr_scale: float = float(
            edge_conf.get("stage2_lr_scale", 0.5)
        )
        self._edge_stage2_lr_scaled: bool = False
        self._edge_warned_no_control: bool = False
        self._edge_last_loss: float = 0.0

    def hook_before_train_loop(self):
        self._write_edge_lock_summary()
        super().hook_before_train_loop()

    def hook_train_loop(self, batch):
        self._maybe_activate_stage2_lr_scale()
        return super().hook_train_loop(batch)

    def _write_edge_lock_summary(self):
        try:
            summary = OrderedDict(
                enabled=self.edge_lock_enabled,
                lambda_edge=self.edge_loss_weight,
                warmup_ratio=self.edge_warmup_ratio,
                every=self.edge_every,
                min_t=self.edge_min_t,
                sample_fraction=self.edge_sample_fraction,
                max_samples=self.edge_max_samples,
                grad_threshold=self.edge_grad_threshold,
                normalize_mode=self.edge_normalize_mode,
                stage2_lr_scale=self.edge_stage2_lr_scale,
            )
            os.makedirs(self.save_root, exist_ok=True)
            path = os.path.join(self.save_root, "edge_lock_summary.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _warmup_steps(self) -> int:
        return int(round(float(self.train_config.steps) * self.edge_warmup_ratio))

    def _in_stage2(self) -> bool:
        return self.step_num >= (self.start_step + self._warmup_steps())

    def _maybe_activate_stage2_lr_scale(self):
        if (
            not self.edge_lock_enabled
            or self.edge_stage2_lr_scale == 1.0
            or self._edge_stage2_lr_scaled
            or not self._in_stage2()
            or self.optimizer is None
        ):
            return

        for group in self.optimizer.param_groups:
            group["lr"] = float(group["lr"]) * self.edge_stage2_lr_scale
        self._edge_stage2_lr_scaled = True
        print_acc(
            f"[alpha_lora_edge_lock] Entered stage 2 at step {self.step_num}; "
            f"scaled optimizer lr by {self.edge_stage2_lr_scale:.4f}"
        )

    def calculate_loss(
        self,
        noise_pred: torch.Tensor,
        noise: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        batch,
        mask_multiplier=1.0,
        prior_pred=None,
        **kwargs,
    ):
        base_loss = super().calculate_loss(
            noise_pred=noise_pred,
            noise=noise,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            batch=batch,
            mask_multiplier=mask_multiplier,
            prior_pred=prior_pred,
            **kwargs,
        )
        self._edge_last_loss = 0.0
        if not self.edge_lock_enabled or self.edge_loss_weight <= 0.0:
            return base_loss
        if not self._in_stage2():
            return base_loss
        if (self.step_num - self.start_step) % self.edge_every != 0:
            return base_loss

        edge_loss = self._compute_edge_locked_loss(
            noise_pred=noise_pred,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            batch=batch,
        )
        if edge_loss is None:
            return base_loss

        self._edge_last_loss = float(edge_loss.detach().item())
        return base_loss + (self.edge_loss_weight * edge_loss)

    def _compute_edge_locked_loss(
        self,
        noise_pred: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        batch,
    ) -> Optional[torch.Tensor]:
        if batch.control_tensor is None:
            if not self._edge_warned_no_control:
                print_acc(
                    "[alpha_lora_edge_lock] control_tensor is missing; "
                    "skipping edge-lock loss."
                )
                self._edge_warned_no_control = True
            return None

        denom = float(
            getattr(self.sd.noise_scheduler.config, "num_train_timesteps", 1000) or 1000
        )
        t_norm = timesteps.to(noise_pred.device).to(noise_pred.dtype) / denom
        valid = torch.nonzero(t_norm > self.edge_min_t, as_tuple=False).flatten()
        if valid.numel() == 0:
            return None

        if self.edge_sample_fraction < 1.0:
            keep = max(1, int(round(valid.numel() * self.edge_sample_fraction)))
        else:
            keep = int(valid.numel())
        if self.edge_max_samples > 0:
            keep = min(keep, self.edge_max_samples)
        if valid.numel() > keep:
            perm = torch.randperm(valid.numel(), device=valid.device)[:keep]
            valid = valid[perm]

        tv = t_norm[valid]
        while len(tv.shape) < len(noise_pred[valid].shape):
            tv = tv.unsqueeze(-1)
        tv = torch.clamp(tv, min=1e-3)
        pred_clean_latents = noisy_latents[valid] - tv * noise_pred[valid]

        pred_imgs = self.sd.decode_latents(
            pred_clean_latents,
            device=noise_pred.device,
            dtype=noise_pred.dtype,
        )
        pred_imgs = ((pred_imgs.float().clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)

        control_imgs = batch.control_tensor[valid].to(
            pred_imgs.device, dtype=pred_imgs.dtype
        )
        if control_imgs.shape[-2:] != pred_imgs.shape[-2:]:
            control_imgs = F.interpolate(
                control_imgs,
                size=pred_imgs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        control_imgs = control_imgs.clamp(0.0, 1.0)

        pred_for_edge = pred_imgs
        control_for_edge = control_imgs
        if self.edge_normalize_mode == "rgb_luma_ratio":
            pred_for_edge = self._normalize_rgb_luma_ratio(pred_imgs, control_imgs)
        elif self.edge_normalize_mode not in ("none", ""):
            if not self._edge_warned_no_control:
                print_acc(
                    f"[alpha_lora_edge_lock] unknown normalize_mode="
                    f"{self.edge_normalize_mode!r}; fallback to none."
                )
                self._edge_warned_no_control = True

        pred_grad = self._log_luma_grad(pred_for_edge)
        control_grad = self._log_luma_grad(control_for_edge)

        pred_mag = torch.sqrt(
            pred_grad[:, 0:1].pow(2) + pred_grad[:, 1:2].pow(2) + self.edge_cos_eps
        )
        control_mag = torch.sqrt(
            control_grad[:, 0:1].pow(2)
            + control_grad[:, 1:2].pow(2)
            + self.edge_cos_eps
        )
        edge_mask = (control_mag > self.edge_grad_threshold).to(pred_imgs.dtype)
        if float(edge_mask.sum().detach().item()) <= 0.0:
            return None

        dot = (pred_grad * control_grad).sum(dim=1, keepdim=True)
        cos = dot / (pred_mag * control_mag + self.edge_cos_eps)
        cos = torch.clamp(cos, min=-1.0, max=1.0)
        loss_map = 1.0 - cos
        return (loss_map * edge_mask).sum() / (edge_mask.sum() + self.edge_cos_eps)

    def _log_luma_grad(self, images_01: torch.Tensor) -> torch.Tensor:
        if images_01.shape[1] == 1:
            luma = images_01
        else:
            luma = (
                0.2126 * images_01[:, 0:1]
                + 0.7152 * images_01[:, 1:2]
                + 0.0722 * images_01[:, 2:3]
            )
        log_luma = torch.log(luma.clamp_min(self.edge_log_eps))
        sobel_x = torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
            device=images_01.device,
            dtype=images_01.dtype,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
            device=images_01.device,
            dtype=images_01.dtype,
        ).view(1, 1, 3, 3)
        grad_x = F.conv2d(log_luma, sobel_x, padding=1)
        grad_y = F.conv2d(log_luma, sobel_y, padding=1)
        return torch.cat([grad_x, grad_y], dim=1)

    def _rgb_luma(self, images_01: torch.Tensor) -> torch.Tensor:
        if images_01.shape[1] == 1:
            return images_01
        return (
            0.2126 * images_01[:, 0:1]
            + 0.7152 * images_01[:, 1:2]
            + 0.0722 * images_01[:, 2:3]
        )

    def _normalize_rgb_luma_ratio(
        self, source_imgs: torch.Tensor, ref_imgs: torch.Tensor
    ) -> torch.Tensor:
        src_luma = self._rgb_luma(source_imgs).clamp_min(self.edge_log_eps)
        ref_luma = self._rgb_luma(ref_imgs).clamp_min(self.edge_log_eps)
        scale = ref_luma / src_luma
        return (source_imgs * scale).clamp(0.0, 1.0)
