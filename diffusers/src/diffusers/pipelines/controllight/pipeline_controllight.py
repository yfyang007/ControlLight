from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from ...utils import replace_example_docstring
from ..flux2.pipeline_flux2_klein import Flux2KleinPipeline
from .pipeline_output import ControlLightPipelineOutput


DEFAULT_PROMPT = (
    "Enhance this low-light image by lifting exposure and recovering visible "
    "details while preserving identity, geometry, atmosphere, natural colors, "
    "and avoiding halos, noise, over-sharpening, or overexposure."
)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import ControlLightPipeline

        >>> pipe = ControlLightPipeline.from_pretrained(
        ...     "/data/yfyang/hf_model/FLUX.2-klein-base-9B",
        ...     torch_dtype=torch.bfloat16,
        ...     default_lora_path="/path/to/controllight_1100.safetensors",
        ... )
        >>> pipe.enable_model_cpu_offload(device="cuda")
        >>> image = pipe(image="input.png", alpha=0.54, num_inference_steps=20).images[0]
        >>> image.save("output.png")
        ```
"""


class ControlLightPipeline(Flux2KleinPipeline):
    def __init__(
        self,
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        transformer,
        is_distilled: bool = False,
        default_lora_path: str | None = None,
        default_adapter_name: str = "controllight",
        default_prompt: str = DEFAULT_PROMPT,
        default_num_inference_steps: int = 20,
        default_guidance_scale: float = 1.0,
        default_max_sequence_length: int = 512,
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            is_distilled=is_distilled,
        )
        self.register_to_config(
            default_lora_path=default_lora_path,
            default_adapter_name=default_adapter_name,
            default_prompt=default_prompt,
            default_num_inference_steps=int(default_num_inference_steps),
            default_guidance_scale=float(default_guidance_scale),
            default_max_sequence_length=int(default_max_sequence_length),
        )
        self._controllight_adapter_loaded = False
        self._controllight_loaded_path: str | None = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, *args, **kwargs):
        pipe = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        pipe._ensure_default_adapter_loaded(pretrained_model_name_or_path)
        return pipe

    @classmethod
    def from_controllight_sources(
        cls,
        model_path: str,
        lora_path: str,
        **kwargs,
    ) -> "ControlLightPipeline":
        kwargs.setdefault("default_lora_path", lora_path)
        return cls.from_pretrained(model_path, **kwargs)

    def _resolve_lora_path(self, source_root: str | Path | None = None) -> str | None:
        raw_path = getattr(self.config, "default_lora_path", None)
        if not raw_path:
            return None
        path = Path(raw_path)
        if path.is_absolute():
            return str(path)
        if source_root is not None:
            candidate = Path(source_root) / path
            if candidate.exists():
                return str(candidate.resolve())
        return str(path)

    def _ensure_default_adapter_loaded(self, source_root: str | Path | None = None) -> None:
        lora_path = self._resolve_lora_path(source_root)
        adapter_name = getattr(self.config, "default_adapter_name", "controllight")
        if not lora_path:
            return
        if self._controllight_adapter_loaded and self._controllight_loaded_path == lora_path:
            return
        self._disable_peft_torchao_dispatch_if_needed()
        self.load_lora_weights(lora_path, adapter_name=adapter_name)
        self.set_adapters(adapter_name, adapter_weights=1.0)
        self._controllight_adapter_loaded = True
        self._controllight_loaded_path = lora_path

    @staticmethod
    def _disable_peft_torchao_dispatch_if_needed() -> None:
        try:
            import peft.import_utils
            import peft.tuners.lora.torchao
        except Exception:
            return

        def _return_false():
            return False

        try:
            peft.import_utils.is_torchao_available.cache_clear()
        except Exception:
            pass

        peft.import_utils.is_torchao_available = _return_false
        peft.tuners.lora.torchao.is_torchao_available = _return_false

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image=None,
        prompt: str | list[str] | None = None,
        alpha: float = 1.0,
        seed: int | None = 42,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        max_sequence_length: int | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        **kwargs: Any,
    ) -> ControlLightPipelineOutput:
        """
        Run single-image ControlLight restoration with a FLUX.2-Klein base model and one LoRA adapter.

        Examples:
        """
        self._ensure_default_adapter_loaded()
        adapter_name = getattr(self.config, "default_adapter_name", "controllight")
        self.set_adapters(adapter_name, adapter_weights=float(alpha))

        if prompt is None:
            prompt = getattr(self.config, "default_prompt", DEFAULT_PROMPT)
        if num_inference_steps is None:
            num_inference_steps = int(getattr(self.config, "default_num_inference_steps", 20))
        if guidance_scale is None:
            guidance_scale = float(getattr(self.config, "default_guidance_scale", 1.0))
        if max_sequence_length is None:
            max_sequence_length = int(getattr(self.config, "default_max_sequence_length", 512))
        if generator is None and seed is not None:
            gen_device = self._execution_device if str(self._execution_device).startswith("cuda") else torch.device("cpu")
            generator = torch.Generator(device=gen_device).manual_seed(int(seed))

        output = super().__call__(
            image=image,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            max_sequence_length=max_sequence_length,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            **kwargs,
        )
        return ControlLightPipelineOutput(images=output.images)


ControlLightDiffusionPipeline = ControlLightPipeline
