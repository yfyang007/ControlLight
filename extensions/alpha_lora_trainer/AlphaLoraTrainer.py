import os
import time
import traceback
import json
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from PIL.ImageOps import exif_transpose

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.config_modules import GenerateImageConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.print import print_acc
from extensions_built_in.diffusion_models.flux2.src.sampling import (
    batched_prc_img,
    get_schedule,
)


DEFAULT_ALPHA_PROMPT = (
    "Enhance this low-light image by lifting exposure and recovering visible "
    "details while preserving identity, geometry, atmosphere, natural colors, "
    "and avoiding halos, noise, over-sharpening, or overexposure."
)


class _AlphaScheduleTransformer(torch.nn.Module):
    def __init__(self, base_transformer, network, alphas_per_step: List[float]):
        super().__init__()
        self.base_transformer = base_transformer
        self.network = network
        self.alphas_per_step = [float(x) for x in alphas_per_step]
        self.step_idx = 0
        self.start_multiplier = float(getattr(network, "multiplier", 1.0))

    def forward(self, *args, **kwargs):
        idx = min(self.step_idx, len(self.alphas_per_step) - 1)
        self.network.multiplier = float(self.alphas_per_step[idx])
        self.step_idx += 1
        return self.base_transformer(*args, **kwargs)

    def restore(self):
        self.network.multiplier = self.start_multiplier
        if hasattr(self.network, "_update_torch_multiplier"):
            self.network._update_torch_multiplier()

    def __getattr__(self, name):
        if name in {
            "base_transformer",
            "network",
            "alphas_per_step",
            "step_idx",
            "start_multiplier",
        }:
            return super().__getattr__(name)
        return getattr(self.base_transformer, name)


class AlphaLoraTrainer(SDTrainer):
    """
    SDTrainer variant for alpha-conditioned LoRA training.

    ai-toolkit already routes DatasetConfig.network_weight into the LoRA
    multiplier used during each forward pass, so the training alpha is simply
    each sample's `network_weight`. This class keeps that math unchanged and
    adds periodic current-batch alpha sweeps:

        source | target | pred alpha=0 | pred alpha=0.5 | pred alpha=1

    Images are saved under <save_root>/<output_dir> and optionally logged to
    TensorBoard when `log_dir` is configured.
    """

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        vis_config = self.get_conf("alpha_visualization", {}) or {}

        self.alpha_vis_enabled: bool = bool(vis_config.get("enabled", True))
        self.alpha_vis_only: bool = bool(vis_config.get("visualize_only", False))
        self.alpha_vis_every: int = int(vis_config.get("every", 100))
        self.alpha_vis_alphas: List[float] = [
            float(x) for x in vis_config.get("alphas", [0.0, 0.5, 1.0])
        ]
        self.alpha_vis_random_middle: bool = bool(
            vis_config.get("random_middle_alpha", False)
        )
        self.alpha_vis_random_middle_min: float = float(
            vis_config.get("random_middle_alpha_min", 0.0)
        )
        self.alpha_vis_random_middle_max: float = float(
            vis_config.get("random_middle_alpha_max", 1.0)
        )
        self.alpha_vis_before_first_step: bool = bool(
            vis_config.get("before_first_step", False)
        )
        self._alpha_vis_did_before_first_step: bool = False
        # Periodic generation inside a DDP train step must not make peer ranks
        # wait in torch/NCCL collectives while the main rank is sampling. When
        # enabled, peer ranks wait on a filesystem sentinel instead; this keeps
        # all ranks aligned without creating a long-running NCCL work item that
        # the watchdog can time out.
        self.alpha_vis_distributed_periodic: bool = bool(
            vis_config.get("distributed_periodic", False)
        )
        self.alpha_vis_max_items: int = int(vis_config.get("max_items", 1))
        self.alpha_vis_width: int = int(
            vis_config.get("width", getattr(self.sample_config, "width", 768))
        )
        self.alpha_vis_height: int = int(
            vis_config.get("height", getattr(self.sample_config, "height", 768))
        )
        # For paired edit/enhancement tasks the most useful diagnostic is whether
        # the prediction stays spatially aligned with the current training crop.
        # When enabled, per-batch visualizations use file_item.crop_width/height
        # instead of the fixed sample width/height.
        self.alpha_vis_match_batch_resolution: bool = bool(
            vis_config.get("match_batch_resolution", True)
        )
        self.alpha_vis_tile_width: int = int(vis_config.get("tile_width", 256))
        self.alpha_vis_tile_height: int = int(vis_config.get("tile_height", 256))
        self.alpha_vis_sample_steps: int = int(
            vis_config.get(
                "sample_steps", getattr(self.sample_config, "sample_steps", 20)
            )
        )
        self.alpha_vis_guidance_scale: float = float(
            vis_config.get(
                "guidance_scale", getattr(self.sample_config, "guidance_scale", 1.0)
            )
        )
        self.alpha_vis_seed: int = int(
            vis_config.get("seed", getattr(self.sample_config, "seed", 42))
        )
        self.alpha_vis_prompt: Optional[str] = vis_config.get("prompt", None)
        self.alpha_vis_output_dir: str = vis_config.get(
            "output_dir", "samples_batch"
        )
        self.alpha_vis_fail_on_error: bool = bool(
            vis_config.get("fail_on_error", False)
        )
        self.alpha_vis_format: str = vis_config.get("format", "jpg")
        self.alpha_vis_save_gif: bool = bool(vis_config.get("save_gif", False))
        self.alpha_vis_gif_duration_ms: int = int(vis_config.get("gif_duration_ms", 450))
        self.alpha_vis_noise_mode: str = str(
            vis_config.get("noise_mode", "seeded")
        ).strip().lower()
        self.alpha_vis_schedule_specs: List[str] = [
            str(x).strip()
            for x in vis_config.get("schedule_specs", [])
            if str(x).strip()
        ]
        if self.alpha_vis_noise_mode not in {"seeded", "shared"}:
            print_acc(
                f"[alpha_lora_trainer] Unsupported alpha_visualization.noise_mode="
                f"{self.alpha_vis_noise_mode!r}; falling back to 'seeded'."
            )
            self.alpha_vis_noise_mode = "seeded"

    def hook_train_loop(self, batch):
        self._maybe_visualize_alpha_batch(batch, before_first_step=True)
        if self.alpha_vis_only:
            return OrderedDict({"loss": 0.0})
        loss_dict = super().hook_train_loop(batch)
        self._maybe_visualize_alpha_batch(batch)
        return loss_dict

    def _maybe_visualize_alpha_batch(
        self,
        batch,
        before_first_step: bool = False,
    ) -> None:
        if not self.alpha_vis_enabled:
            return
        if before_first_step:
            if (
                not self.alpha_vis_before_first_step
                or self._alpha_vis_did_before_first_step
            ):
                return
        elif self.alpha_vis_every <= 0:
            return
        elif (
            getattr(self.accelerator, "num_processes", 1) > 1
            and not self.alpha_vis_distributed_periodic
        ):
            return
        elif self.step_num == self.start_step:
            return
        elif self.step_num % self.alpha_vis_every != 0:
            return
        if self.network is None:
            return

        num_processes = getattr(self.accelerator, "num_processes", 1)
        distributed_done_path = (
            self._distributed_visualization_done_path()
            if num_processes > 1
            else None
        )
        wait_started_at = time.time()

        try:
            if self.accelerator.is_main_process:
                batch_dto = self._select_batch(batch)
                if batch_dto is None or not batch_dto.file_items:
                    return
                self._visualize_alpha_batch(batch_dto)
        except Exception as exc:
            print_acc("")
            print_acc(
                f"[alpha_lora_trainer] Warning: failed current-batch "
                f"visualization at step {self.step_num}: {exc}"
            )
            print_acc(traceback.format_exc())
            if self.alpha_vis_fail_on_error:
                raise
        finally:
            if before_first_step:
                self._alpha_vis_did_before_first_step = True
            if num_processes > 1 and distributed_done_path is not None:
                if self.accelerator.is_main_process:
                    self._write_distributed_visualization_done(distributed_done_path)
                else:
                    self._wait_distributed_visualization_done(
                        distributed_done_path,
                        wait_started_at,
                    )

    def _distributed_visualization_done_path(self) -> str:
        step_dir = os.path.join(
            self.save_root,
            self.alpha_vis_output_dir,
            f"step_{self.step_num:09d}",
        )
        return os.path.join(step_dir, ".ddp_visualization_done")

    @staticmethod
    def _write_distributed_visualization_done(done_path: str) -> None:
        os.makedirs(os.path.dirname(done_path), exist_ok=True)
        tmp_path = f"{done_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(f"{time.time()}\n")
        os.replace(tmp_path, done_path)

    @staticmethod
    def _wait_distributed_visualization_done(
        done_path: str,
        wait_started_at: float,
    ) -> None:
        timeout = float(os.environ.get("ALPHA_VIS_DDP_WAIT_TIMEOUT_SEC", "0") or 0)
        while True:
            try:
                if os.path.getmtime(done_path) >= wait_started_at - 1.0:
                    return
            except FileNotFoundError:
                pass
            if timeout > 0 and time.time() - wait_started_at > timeout:
                raise TimeoutError(
                    f"Timed out waiting for DDP visualization sentinel: {done_path}"
                )
            time.sleep(2)

    @staticmethod
    def _select_batch(batch) -> Optional[DataLoaderBatchDTO]:
        if isinstance(batch, DataLoaderBatchDTO):
            return batch
        if isinstance(batch, list):
            for item in batch:
                if isinstance(item, DataLoaderBatchDTO):
                    return item
        return None

    def _visualize_alpha_batch(self, batch: DataLoaderBatchDTO) -> None:
        output_dir = os.path.join(self.save_root, self.alpha_vis_output_dir)
        step_dir = os.path.join(output_dir, f"step_{self.step_num:09d}")
        os.makedirs(step_dir, exist_ok=True)

        grids = []
        num_items = min(self.alpha_vis_max_items, len(batch.file_items))
        old_sample_cache = getattr(self.sd, "sample_prompts_cache", None)

        try:
            # Current-batch prompts/control images are not the fixed sample prompts,
            # so disable the fixed sample embedding cache temporarily.
            self.sd.sample_prompts_cache = None

            for item_idx in range(num_items):
                file_item = batch.file_items[item_idx]
                control_paths = self._control_paths_for_item(file_item)
                if not control_paths:
                    print_acc(
                        f"[alpha_lora_trainer] No control image for {file_item.path}; "
                        "skipping batch visualization item."
                    )
                    continue

                prompt = self.alpha_vis_prompt or file_item.caption or DEFAULT_ALPHA_PROMPT
                target_alpha = float(getattr(file_item, "network_weight", 1.0))
                gen_width, gen_height = self._generation_size_for_item(file_item)
                configs = []
                pred_paths = []

                alphas = self._alpha_values_for_step(item_idx)
                shared_latents = self._build_shared_start_latents(
                    width=gen_width,
                    height=gen_height,
                    seed=self.alpha_vis_seed + item_idx,
                )

                use_schedule = len(self.alpha_vis_schedule_specs) > 0
                schedule_labels: List[str] = []
                schedule_summary = []

                if use_schedule:
                    for raw_spec in self.alpha_vis_schedule_specs:
                        spec = self._parse_schedule_spec(raw_spec)
                        pred_path = os.path.join(
                            step_dir,
                            f"item_{item_idx:02d}_{spec['name']}.{self.alpha_vis_format}",
                        )
                        pred_paths.append(pred_path)
                        schedule_labels.append(spec["name"])
                        cfg = GenerateImageConfig(
                            prompt=prompt,
                            width=gen_width,
                            height=gen_height,
                            negative_prompt="",
                            seed=self.alpha_vis_seed + item_idx,
                            guidance_scale=self.alpha_vis_guidance_scale,
                            num_inference_steps=self.alpha_vis_sample_steps,
                            network_multiplier=float(spec["target_alpha"]),
                            output_path=pred_path,
                            output_ext=self.alpha_vis_format,
                            logger=None,
                            ctrl_img_1=control_paths[0] if len(control_paths) > 0 else None,
                            ctrl_img_2=control_paths[1] if len(control_paths) > 1 else None,
                            ctrl_img_3=control_paths[2] if len(control_paths) > 2 else None,
                            latents=(
                                shared_latents.clone()
                                if shared_latents is not None
                                else None
                            ),
                        )
                        used_alphas = self._generate_single_with_schedule(cfg, spec)
                        schedule_summary.append(
                            {
                                "raw": raw_spec,
                                "name": spec["name"],
                                "used_alphas": used_alphas,
                                "output_path": pred_path,
                            }
                        )
                else:
                    for alpha in alphas:
                        pred_path = os.path.join(
                            step_dir,
                            f"item_{item_idx:02d}_alpha_{alpha:.3f}.{self.alpha_vis_format}",
                        )
                        pred_paths.append(pred_path)
                        configs.append(
                            GenerateImageConfig(
                                prompt=prompt,
                                width=gen_width,
                                height=gen_height,
                                negative_prompt="",
                                seed=self.alpha_vis_seed + item_idx,
                                guidance_scale=self.alpha_vis_guidance_scale,
                                num_inference_steps=self.alpha_vis_sample_steps,
                                network_multiplier=alpha,
                                output_path=pred_path,
                                output_ext=self.alpha_vis_format,
                                logger=None,
                                ctrl_img_1=control_paths[0] if len(control_paths) > 0 else None,
                                ctrl_img_2=control_paths[1] if len(control_paths) > 1 else None,
                                ctrl_img_3=control_paths[2] if len(control_paths) > 2 else None,
                                latents=(
                                    shared_latents.clone()
                                    if shared_latents is not None
                                    else None
                                ),
                            )
                        )

                print_acc(
                    f"[alpha_lora_trainer] Visualizing current batch at step "
                    f"{self.step_num}, item {item_idx}, target alpha={target_alpha}, "
                    f"noise_mode={self.alpha_vis_noise_mode}"
                )
                if use_schedule:
                    summary_path = os.path.join(
                        step_dir, f"item_{item_idx:02d}_schedule_summary.json"
                    )
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(schedule_summary, f, indent=2, ensure_ascii=False)
                else:
                    self.sd.generate_images(configs, sampler=self.sample_config.sampler)

                if use_schedule:
                    grid = self._make_schedule_grid(
                        source_path=control_paths[0],
                        target_path=file_item.path,
                        pred_paths=pred_paths,
                        pred_labels=schedule_labels,
                        target_alpha=target_alpha,
                        file_item=file_item,
                    )
                else:
                    grid = self._make_grid(
                        source_path=control_paths[0],
                        target_path=file_item.path,
                        pred_paths=pred_paths,
                        alphas=alphas,
                        target_alpha=target_alpha,
                        file_item=file_item,
                    )
                grid_path = os.path.join(
                    step_dir, f"item_{item_idx:02d}_grid.{self.alpha_vis_format}"
                )
                grid.save(grid_path)
                grids.append((item_idx, grid, target_alpha))
                if self.alpha_vis_save_gif:
                    gif_path = os.path.join(
                        step_dir, f"item_{item_idx:02d}_alpha_sweep.gif"
                    )
                    self._save_alpha_gif(pred_paths, alphas, gif_path)

                if self.writer is not None:
                    self.writer.add_image(
                        f"alpha_batch/item_{item_idx:02d}",
                        np.asarray(grid),
                        self.step_num,
                        dataformats="HWC",
                    )
                    self.writer.add_scalar(
                        f"alpha_batch/item_{item_idx:02d}_target_alpha",
                        target_alpha,
                        self.step_num,
                    )

            if grids and self.writer is not None:
                self.writer.flush()
        finally:
            self.sd.sample_prompts_cache = old_sample_cache

    @staticmethod
    def _control_paths_for_item(file_item) -> List[str]:
        control_path = getattr(file_item, "control_path", None)
        if control_path is None:
            return []
        if isinstance(control_path, list):
            return [str(x) for x in control_path if x is not None]
        return [str(control_path)]

    def _alpha_values_for_step(self, item_idx: int = 0) -> List[float]:
        alphas = list(self.alpha_vis_alphas)
        if self.alpha_vis_random_middle:
            low = min(self.alpha_vis_random_middle_min, self.alpha_vis_random_middle_max)
            high = max(self.alpha_vis_random_middle_min, self.alpha_vis_random_middle_max)
            rng = np.random.default_rng(
                self.alpha_vis_seed + self.step_num * 1009 + item_idx * 9173
            )
            random_alpha = float(rng.uniform(low, high))
            # Round so filenames/grid labels are readable and repeated runs at
            # the same step use exactly the same display value.
            alphas.append(round(random_alpha, 3))

        # Keep the sweep left-to-right and avoid duplicate values if the random
        # draw lands on a configured alpha after rounding.
        return sorted(set(float(x) for x in alphas))

    def _generation_size_for_item(self, file_item) -> tuple[int, int]:
        if self.alpha_vis_match_batch_resolution:
            width = int(getattr(file_item, "crop_width", 0) or 0)
            height = int(getattr(file_item, "crop_height", 0) or 0)
            if width > 0 and height > 0:
                return width, height
        return self.alpha_vis_width, self.alpha_vis_height

    @staticmethod
    def _parse_schedule_spec(raw_spec: str) -> dict:
        pieces = [x.strip() for x in str(raw_spec).split(":")]
        kind = pieces[0].lower()
        if kind == "fixed" and len(pieces) == 2:
            target = float(pieces[1])
            return {
                "kind": kind,
                "name": f"fixed_a{target:.3f}",
                "target_alpha": target,
                "base_alpha": target,
                "hold_fraction": 1.0,
            }
        if kind == "hold_then_ramp" and len(pieces) == 4:
            base = float(pieces[1])
            target = float(pieces[2])
            hold = float(pieces[3])
            return {
                "kind": kind,
                "name": f"hold{base:.3f}_to_{target:.3f}_h{hold:.2f}",
                "target_alpha": target,
                "base_alpha": base,
                "hold_fraction": hold,
            }
        raise ValueError(f"Unsupported schedule spec: {raw_spec}")

    @staticmethod
    def _alphas_for_schedule(spec: dict, num_pairs: int) -> List[float]:
        if spec["kind"] == "fixed":
            return [float(spec["target_alpha"]) for _ in range(max(num_pairs, 1))]
        hold_steps = int(round(float(spec["hold_fraction"]) * max(num_pairs, 1)))
        hold_steps = max(0, min(hold_steps, max(num_pairs, 1)))
        values: List[float] = []
        for step_idx in range(max(num_pairs, 1)):
            if step_idx < hold_steps:
                values.append(float(spec["base_alpha"]))
            else:
                ramp_len = max(max(num_pairs, 1) - hold_steps, 1)
                progress = min(max((step_idx - hold_steps + 1) / ramp_len, 0.0), 1.0)
                values.append(
                    float(
                        spec["base_alpha"]
                        + (spec["target_alpha"] - spec["base_alpha"]) * progress
                    )
                )
        return values

    def _build_shared_start_latents(
        self,
        width: int,
        height: int,
        seed: int,
    ):
        if self.alpha_vis_noise_mode != "shared":
            return None
        pipeline = getattr(self.sd, "pipeline", None)
        if pipeline is None or not hasattr(pipeline, "prepare_latents"):
            return None

        device = torch.device(self.sd.device_torch)
        generator_device = device if device.type == "cuda" else torch.device("cpu")
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        num_channels_latents = getattr(pipeline, "num_channels_latents", None)
        if num_channels_latents is None:
            transformer = (
                getattr(pipeline, "transformer", None)
                or getattr(self.sd, "transformer", None)
                or getattr(self.sd, "unet", None)
            )
            transformer_cfg = getattr(transformer, "config", None)
            num_channels_latents = int(
                getattr(transformer_cfg, "in_channels", 0) or 0
            ) or 4

        prepared = pipeline.prepare_latents(
            1,
            int(num_channels_latents),
            int(height),
            int(width),
            self.sd.torch_dtype,
            device,
            generator,
            None,
        )
        latents = prepared[0] if isinstance(prepared, tuple) else prepared
        return latents.detach().clone()

    def _generate_single_with_schedule(self, gen_config: GenerateImageConfig, spec: dict) -> List[float]:
        if self.sd.network is None:
            raise ValueError("Schedule visualization requires self.sd.network")
        pipeline = self.sd.get_generation_pipeline()
        try:
            pipeline.set_progress_bar_config(disable=True)
        except Exception:
            pass

        if gen_config.latents is not None:
            latents = gen_config.latents.clone()
        else:
            latents = pipeline.prepare_latents(
                1,
                pipeline.num_channels_latents,
                gen_config.height,
                gen_config.width,
                self.sd.torch_dtype,
                torch.device(self.sd.device_torch),
                torch.Generator(
                    device=(
                        torch.device(self.sd.device_torch)
                        if torch.device(self.sd.device_torch).type == "cuda"
                        else torch.device("cpu")
                    )
                ).manual_seed(int(gen_config.seed)),
                None,
            )
        packed_latents, _ = batched_prc_img(latents)
        timesteps = get_schedule(gen_config.num_inference_steps, packed_latents.shape[1])
        alphas_per_step = self._alphas_for_schedule(spec, len(timesteps) - 1)

        wrapped = _AlphaScheduleTransformer(
            base_transformer=pipeline.transformer,
            network=self.sd.network,
            alphas_per_step=alphas_per_step,
        )
        old_transformer = pipeline.transformer
        old_can_merge_in = bool(getattr(self.sd.network, "can_merge_in", False))
        pipeline.transformer = wrapped
        self.sd.network.can_merge_in = False
        try:
            self.sd.generate_images(
                [gen_config],
                sampler=self.sample_config.sampler,
                pipeline=pipeline,
            )
        finally:
            wrapped.restore()
            pipeline.transformer = old_transformer
            self.sd.network.can_merge_in = old_can_merge_in
        return alphas_per_step

    def _save_alpha_gif(
        self,
        pred_paths: List[str],
        alphas: List[float],
        gif_path: str,
    ) -> None:
        frames: List[Image.Image] = []
        for path, alpha in zip(pred_paths, alphas):
            try:
                img = exif_transpose(Image.open(path)).convert("RGB")
            except Exception:
                continue
            frame = img.copy()
            draw = ImageDraw.Draw(frame)
            label = f"alpha={alpha:.3f}"
            # Draw a small high-contrast label so the GIF is self-describing.
            text_bbox = draw.textbbox((0, 0), label)
            pad = 8
            box = (
                8,
                8,
                8 + (text_bbox[2] - text_bbox[0]) + pad * 2,
                8 + (text_bbox[3] - text_bbox[1]) + pad * 2,
            )
            draw.rectangle(box, fill=(0, 0, 0))
            draw.text((8 + pad, 8 + pad), label, fill=(255, 255, 255))
            frames.append(frame)

        if not frames:
            return
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=self.alpha_vis_gif_duration_ms,
            loop=0,
        )

    def _make_grid(
        self,
        source_path: str,
        target_path: str,
        pred_paths: List[str],
        alphas: List[float],
        target_alpha: float,
        file_item=None,
    ) -> Image.Image:
        labels = ["source", f"target α={target_alpha:g}"] + [
            f"pred α={alpha:g}" for alpha in alphas
        ]
        paths = [source_path, target_path] + pred_paths

        tiles = [
            self._load_tile(source_path, labels[0], file_item=file_item, align_to_item=True),
            self._load_tile(target_path, labels[1], file_item=file_item, align_to_item=True),
        ] + [
            self._load_tile(path, label)
            for path, label in zip(pred_paths, labels[2:])
        ]
        grid_w = self.alpha_vis_tile_width * len(tiles)
        grid_h = self.alpha_vis_tile_height + 28
        grid = Image.new("RGB", (grid_w, grid_h), "white")
        draw = ImageDraw.Draw(grid)

        for idx, (tile, label) in enumerate(zip(tiles, labels)):
            x = idx * self.alpha_vis_tile_width
            grid.paste(tile, (x, 0))
            draw.text((x + 6, self.alpha_vis_tile_height + 7), label, fill=(0, 0, 0))

        return grid

    def _make_schedule_grid(
        self,
        source_path: str,
        target_path: str,
        pred_paths: List[str],
        pred_labels: List[str],
        target_alpha: float,
        file_item=None,
    ) -> Image.Image:
        labels = ["source", f"target α={target_alpha:g}"] + list(pred_labels)
        paths = [source_path, target_path] + pred_paths
        tiles = [
            self._load_tile(source_path, labels[0], file_item=file_item, align_to_item=True),
            self._load_tile(target_path, labels[1], file_item=file_item, align_to_item=True),
        ] + [
            self._load_tile(path, label)
            for path, label in zip(pred_paths, labels[2:])
        ]
        grid_w = self.alpha_vis_tile_width * len(tiles)
        grid_h = self.alpha_vis_tile_height + 28
        grid = Image.new("RGB", (grid_w, grid_h), "white")
        draw = ImageDraw.Draw(grid)
        for idx, (tile, label) in enumerate(zip(tiles, labels)):
            x = idx * self.alpha_vis_tile_width
            grid.paste(tile, (x, 0))
            draw.text((x + 6, self.alpha_vis_tile_height + 7), label, fill=(0, 0, 0))
        return grid

    def _load_tile(
        self,
        path: str,
        label: str,
        file_item=None,
        align_to_item: bool = False,
    ) -> Image.Image:
        try:
            img = exif_transpose(Image.open(path)).convert("RGB")
            if align_to_item and file_item is not None:
                img = self._apply_file_item_spatial_transform(img, file_item)
        except Exception:
            img = Image.new(
                "RGB",
                (self.alpha_vis_tile_width, self.alpha_vis_tile_height),
                (180, 40, 40),
            )
            draw = ImageDraw.Draw(img)
            draw.text((8, 8), f"missing\n{label}", fill=(255, 255, 255))
            return img

        img.thumbnail(
            (self.alpha_vis_tile_width, self.alpha_vis_tile_height),
            Image.Resampling.LANCZOS,
        )
        canvas = Image.new(
            "RGB",
            (self.alpha_vis_tile_width, self.alpha_vis_tile_height),
            (245, 245, 245),
        )
        x = (self.alpha_vis_tile_width - img.width) // 2
        y = (self.alpha_vis_tile_height - img.height) // 2
        canvas.paste(img, (x, y))
        return canvas

    @staticmethod
    def _apply_file_item_spatial_transform(img: Image.Image, file_item) -> Image.Image:
        """Apply the same flip/resize/crop metadata used by the bucket loader."""
        if getattr(file_item, "flip_x", False):
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if getattr(file_item, "flip_y", False):
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        scale_to_width = int(getattr(file_item, "scale_to_width", img.width) or img.width)
        scale_to_height = int(getattr(file_item, "scale_to_height", img.height) or img.height)
        crop_x = int(getattr(file_item, "crop_x", 0) or 0)
        crop_y = int(getattr(file_item, "crop_y", 0) or 0)
        crop_width = int(getattr(file_item, "crop_width", scale_to_width) or scale_to_width)
        crop_height = int(getattr(file_item, "crop_height", scale_to_height) or scale_to_height)

        if scale_to_width > 0 and scale_to_height > 0:
            img = img.resize((scale_to_width, scale_to_height), Image.Resampling.BICUBIC)

        if crop_width > 0 and crop_height > 0:
            img = img.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))

        return img
