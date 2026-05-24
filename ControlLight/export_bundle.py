from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_DIFFUSERS_SRC = _REPO_ROOT / "diffusers" / "src"
if str(_LOCAL_DIFFUSERS_SRC) not in sys.path:
    sys.path.insert(0, str(_LOCAL_DIFFUSERS_SRC))

import torch


DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def export_bundle_from_source(
    model_path: str,
    lora_path: str,
    save_dir: str,
    device: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    safe_serialization: bool = True,
    default_prompt: str | None = None,
) -> Path:
    from diffusers import ControlLightPipeline

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    local_lora_dir = save_path / "lora"
    local_lora_dir.mkdir(parents=True, exist_ok=True)
    local_lora_path = local_lora_dir / Path(lora_path).name
    shutil.copy2(lora_path, local_lora_path)

    pipe = ControlLightPipeline.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        default_lora_path=str(local_lora_path),
        default_prompt=default_prompt,
    )
    pipe.register_to_config(default_lora_path=f"lora/{local_lora_path.name}")
    pipe.to("cpu")
    pipe.save_pretrained(save_path, safe_serialization=safe_serialization)
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a self-contained ControlLight diffusers bundle."
    )
    parser.add_argument("--model_path", required=True, help="Base FLUX.2-Klein model path.")
    parser.add_argument("--lora_path", required=True, help="ControlLight LoRA weights path.")
    parser.add_argument("--save_dir", required=True, help="Output bundle directory.")
    parser.add_argument("--device", default="cuda", help="Device used while assembling the bundle.")
    parser.add_argument(
        "--torch_dtype",
        default="bfloat16",
        choices=sorted(DTYPE_MAP),
        help="Load dtype for the temporary assembly pipeline.",
    )
    parser.add_argument(
        "--disable_safe_serialization",
        action="store_true",
        help="Save weights with PyTorch binaries instead of safetensors.",
    )
    parser.add_argument(
        "--default_prompt",
        default=None,
        help="Optional default prompt to bake into the bundle config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_path = export_bundle_from_source(
        model_path=args.model_path,
        lora_path=args.lora_path,
        save_dir=args.save_dir,
        device=args.device,
        torch_dtype=DTYPE_MAP[args.torch_dtype],
        safe_serialization=not args.disable_safe_serialization,
        default_prompt=args.default_prompt,
    )
    print(f"Saved ControlLight diffusers bundle to {save_path}")


if __name__ == "__main__":
    main()
