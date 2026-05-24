import json
import os
import random
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from extensions.alpha_lora_trainer.AlphaLoraTrainer import AlphaLoraTrainer
from .pair_dataloader import build_same_sample_pair_dataloader_from_existing


class AlphaLoraStructConsistencyTrainer(AlphaLoraTrainer):
    """
    Alpha-conditioned LoRA trainer with same-sample structural consistency loss.

    For each batch, the custom dataloader emits same-input / different-strength
    pairs. We decode a small random subset of predicted x0 latents, apply
    per-channel color-map normalization, then optimize a Sobel edge consistency
    loss across the paired predictions.
    """

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        sc_conf = self.get_conf("structure_consistency", {}) or {}
        self.sc_enabled: bool = bool(sc_conf.get("enabled", True))
        self.sc_loss_weight: float = float(sc_conf.get("loss_weight", 0.05))
        self.sc_pairs_per_batch: int = int(sc_conf.get("pairs_per_batch", 1))
        self.sc_color_map_eps: float = float(sc_conf.get("color_map_eps", 1e-6))
        self.sc_sobel_eps: float = float(sc_conf.get("sobel_eps", 1e-6))
        self.sc_share_spatial_crop: bool = bool(
            sc_conf.get("share_spatial_crop", True)
        )
        self.sc_share_noise_within_pair: bool = bool(
            sc_conf.get("share_noise_within_pair", True)
        )
        self.sc_share_timestep_within_pair: bool = bool(
            sc_conf.get("share_timestep_within_pair", True)
        )
        self._sc_loader_replaced: bool = False
        self._sc_last_loss: float = 0.0

    def hook_before_train_loop(self):
        if self.sc_enabled and not self._sc_loader_replaced and self.data_loader is not None:
            self.data_loader = build_same_sample_pair_dataloader_from_existing(
                self.data_loader,
                batch_size=self.train_config.batch_size,
                share_spatial_crop=self.sc_share_spatial_crop,
                seed=getattr(self.sample_config, "seed", 42),
            )
            self._sc_loader_replaced = True
            self._write_structure_consistency_summary()
        super().hook_before_train_loop()

    def _write_structure_consistency_summary(self):
        try:
            dataset = self.data_loader.dataset
            summary = OrderedDict(
                enabled=self.sc_enabled,
                loss_weight=self.sc_loss_weight,
                pairs_per_batch=self.sc_pairs_per_batch,
                share_spatial_crop=self.sc_share_spatial_crop,
                share_noise_within_pair=self.sc_share_noise_within_pair,
                share_timestep_within_pair=self.sc_share_timestep_within_pair,
                grouped_sample_count=len(getattr(dataset, "group_keys", [])),
                wrapped_dataset_count=len(getattr(dataset, "datasets", [])),
            )
            path = os.path.join(self.save_root, "structure_consistency_summary.json")
            os.makedirs(self.save_root, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def process_general_training_batch(self, batch):
        noisy_latents, noise, timesteps, conditioned_prompts, imgs = super().process_general_training_batch(batch)
        if not self.sc_enabled:
            return noisy_latents, noise, timesteps, conditioned_prompts, imgs

        pair_list = getattr(batch, "consistency_pairs", None)
        if not pair_list:
            return noisy_latents, noise, timesteps, conditioned_prompts, imgs

        if not (self.sc_share_noise_within_pair or self.sc_share_timestep_within_pair):
            return noisy_latents, noise, timesteps, conditioned_prompts, imgs

        noise = noise.clone()
        timesteps = timesteps.clone()
        for left_idx, right_idx in pair_list:
            if self.sc_share_noise_within_pair:
                noise[right_idx] = noise[left_idx]
            if self.sc_share_timestep_within_pair:
                timesteps[right_idx] = timesteps[left_idx]

        latents = batch.latents
        noisy_latents = self.sd.add_noise(latents, noise, timesteps)
        if self.train_config.loss_target in ["source", "unaugmented"]:
            batch.sigmas = self.get_sigmas(
                timesteps, len(noisy_latents.shape), noisy_latents.dtype
            )

        noisy_latent_multiplier = self.train_config.noisy_latent_multiplier
        if noisy_latent_multiplier != 1.0:
            noisy_latents = noisy_latents * noisy_latent_multiplier

        noisy_latents.requires_grad = False
        noisy_latents = noisy_latents.detach()
        noise.requires_grad = False
        noise = noise.detach()
        return noisy_latents, noise, timesteps, conditioned_prompts, imgs

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
        self._sc_last_loss = 0.0
        if not self.sc_enabled or self.sc_loss_weight <= 0.0:
            return base_loss

        sc_loss = self._compute_structure_consistency_loss(
            noise_pred=noise_pred,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            batch=batch,
        )
        if sc_loss is None:
            return base_loss

        self._sc_last_loss = float(sc_loss.detach().item())
        return base_loss + (self.sc_loss_weight * sc_loss)

    def _compute_structure_consistency_loss(
        self,
        noise_pred: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        batch,
    ) -> Optional[torch.Tensor]:
        pair_list: List[Tuple[int, int]] = getattr(batch, "consistency_pairs", None)
        if not pair_list:
            return None

        valid_pairs = [
            (left_idx, right_idx)
            for left_idx, right_idx in pair_list
            if left_idx < noise_pred.shape[0] and right_idx < noise_pred.shape[0]
        ]
        if not valid_pairs:
            return None

        num_pairs = min(len(valid_pairs), max(1, self.sc_pairs_per_batch))
        selected_pairs = random.sample(valid_pairs, num_pairs)
        selected_indices = [idx for pair in selected_pairs for idx in pair]

        x0_pred = self._predict_x0_latents(
            noise_pred=noise_pred[selected_indices],
            noisy_latents=noisy_latents[selected_indices],
            timesteps=timesteps[selected_indices],
        )
        decoded = self.sd.decode_latents(
            x0_pred,
            device=noise_pred.device,
            dtype=noise_pred.dtype,
        )
        decoded = ((decoded.float().clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)
        color_map = self._color_map_normalize(decoded)
        sobel_mag = self._sobel_magnitude(color_map)

        pair_losses = []
        for pair_offset in range(num_pairs):
            left = sobel_mag[pair_offset * 2]
            right = sobel_mag[pair_offset * 2 + 1]
            pair_losses.append(F.l1_loss(left, right))

        if len(pair_losses) == 0:
            return None
        return torch.stack(pair_losses).mean()

    def _predict_x0_latents(
        self,
        noise_pred: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        tv = timesteps.to(noise_pred.device).to(noise_pred.dtype) / 1000.0
        while len(tv.shape) < len(noise_pred.shape):
            tv = tv.unsqueeze(-1)
        tv = torch.clamp(tv, min=1e-3)
        return noisy_latents - tv * noise_pred

    def _color_map_normalize(self, images_01: torch.Tensor) -> torch.Tensor:
        channel_max = images_01.amax(dim=(-2, -1), keepdim=True)
        channel_max = channel_max.clamp_min(self.sc_color_map_eps)
        return images_01 / channel_max

    def _sobel_magnitude(self, images: torch.Tensor) -> torch.Tensor:
        channels = images.shape[1]
        sobel_x = torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
            device=images.device,
            dtype=images.dtype,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
            device=images.device,
            dtype=images.dtype,
        ).view(1, 1, 3, 3)
        sobel_x = sobel_x.repeat(channels, 1, 1, 1)
        sobel_y = sobel_y.repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(images, sobel_x, padding=1, groups=channels)
        grad_y = F.conv2d(images, sobel_y, padding=1, groups=channels)
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + self.sc_sobel_eps)
