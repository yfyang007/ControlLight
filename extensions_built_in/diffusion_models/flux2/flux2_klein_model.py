from .flux2_model import Flux2Model
import os
from transformers import Qwen3ForCausalLM, AutoTokenizer
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype
from toolkit.config_modules import ModelConfig
from toolkit.memory_management.manager import MemoryManager
from toolkit.basic import flush
from .src.model import Klein9BParams, Klein4BParams


class Flux2KleinModel(Flux2Model):
    flux2_klein_te_path: str = None
    flux2_te_type: str = "qwen"  # "mistral" or "qwen"
    flux2_vae_path: str = "ai-toolkit/flux2_vae"
    flux2_is_guidance_distilled: bool = False

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device,
            model_config,
            dtype,
            custom_pipeline,
            noise_scheduler,
            **kwargs,
        )
        # use the new format on this new model by default
        self.use_old_lokr_format = False

    def load_te(self):
        te_path = self.model_config.te_name_or_path
        tokenizer_path = te_path

        # Diffusers FLUX.2 Klein repos store Qwen3 in subfolders:
        #   <repo>/text_encoder
        #   <repo>/tokenizer
        # The previous edit scripts use Flux2KleinPipeline.from_pretrained(repo),
        # so Qwen3 was loaded from those bundled subfolders rather than from a
        # separate Qwen/Qwen3-* checkout. Mirror that behavior here.
        if te_path is None and os.path.isdir(self.model_config.name_or_path):
            bundled_te_path = os.path.join(self.model_config.name_or_path, "text_encoder")
            bundled_tokenizer_path = os.path.join(self.model_config.name_or_path, "tokenizer")
            if os.path.exists(os.path.join(bundled_te_path, "config.json")):
                te_path = bundled_te_path
                if os.path.isdir(bundled_tokenizer_path):
                    tokenizer_path = bundled_tokenizer_path

        if te_path is None:
            te_path = self.flux2_klein_te_path
            tokenizer_path = te_path

        if tokenizer_path is None:
            tokenizer_path = te_path
        if te_path is None:
            raise ValueError("flux2_klein_te_path must be set for Flux2KleinModel")
        dtype = self.torch_dtype
        self.print_and_status_update(f"Loading Qwen3 from {te_path}")

        text_encoder: Qwen3ForCausalLM = Qwen3ForCausalLM.from_pretrained(
            te_path,
            torch_dtype=dtype,
        )
        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Qwen3")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()
        elif not self.model_config.low_vram:
            text_encoder.to(self.device_torch, dtype=dtype)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return text_encoder, tokenizer


class Flux2Klein4BModel(Flux2KleinModel):
    arch = "flux2_klein_4b"
    flux2_klein_te_path: str = "Qwen/Qwen3-4B"
    flux2_te_filename: str = "flux-2-klein-base-4b.safetensors"

    def get_flux2_params(self):
        return Klein4BParams()

    def get_base_model_version(self):
        return "flux2_klein_4b"


class Flux2Klein9BModel(Flux2KleinModel):
    arch = "flux2_klein_9b"
    flux2_klein_te_path: str = "Qwen/Qwen3-8B"
    flux2_te_filename: str = "flux-2-klein-base-9b.safetensors"

    def get_flux2_params(self):
        return Klein9BParams()

    def get_base_model_version(self):
        return "flux2_klein_9b"
