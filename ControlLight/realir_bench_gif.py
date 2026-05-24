from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from ControlLight.bootstrap import bootstrap_local_paths

_REPO_ROOT = bootstrap_local_paths()

import torch
from PIL import Image, ImageDraw, ImageFont


DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_INPUT_DIR = _REPO_ROOT / "datasets" / "RealIR-Bench-lowlight" / "images"
DEFAULT_MODEL_PATH = None
DEFAULT_LORA_PATH = str((_REPO_ROOT / "weights" / "controllight.safetensors").resolve())
DEFAULT_PROMPTS_PATH = _REPO_ROOT / "prompts" / "realir_bench_lowlight_prompts.json"
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "outputs" / "realir_bench_lowlight_prompt_alpha_gif"


@dataclass
class PromptSpec:
    name: str
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ControlLight on RealIR-Bench-lowlight with prompt and alpha sweeps, then save GIF summaries."
    )
    parser.add_argument("--model-path", "--model_path", dest="model_path", default=DEFAULT_MODEL_PATH, help="Base model path. Required.")
    parser.add_argument("--lora-path", "--lora_path", dest="lora_path", default=DEFAULT_LORA_PATH, help="LoRA weights path.")
    parser.add_argument("--input-dir", "--input_dir", dest="input_dir", default=str(DEFAULT_INPUT_DIR), help="Input image directory.")
    parser.add_argument("--output-root", "--output_root", dest="output_root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root directory.")
    parser.add_argument(
        "--prompts-file",
        "--prompts_file",
        dest="prompts_file",
        default=str(DEFAULT_PROMPTS_PATH),
        help="JSON file containing a list of {'name','prompt'} entries.",
    )
    parser.add_argument("--alphas", default="0.25,0.5,0.75,1.0", help="Comma-separated alpha values.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images after sharding.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards.")
    parser.add_argument("--shard_index", type=int, default=0, help="Current shard index.")
    parser.add_argument("--skip-existing", "--skip_existing", dest="skip_existing", action="store_true", help="Skip images with an existing GIF.")
    parser.add_argument("--seed", type=int, default=42, help="Seed reused across the sweep.")
    parser.add_argument("--num-inference-steps", "--num_inference_steps", dest="num_inference_steps", type=int, default=20, help="Number of denoising steps.")
    parser.add_argument("--guidance-scale", "--guidance_scale", dest="guidance_scale", type=float, default=1.0, help="Guidance scale.")
    parser.add_argument("--max-sequence-length", "--max_sequence_length", dest="max_sequence_length", type=int, default=512, help="Prompt max sequence length.")
    parser.add_argument("--gif-duration-ms", "--gif_duration_ms", dest="gif_duration_ms", type=int, default=450, help="Frame duration in milliseconds.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument(
        "--torch-dtype",
        "--torch_dtype",
        dest="torch_dtype",
        default="bfloat16",
        choices=sorted(DTYPE_MAP),
        help="Inference dtype.",
    )
    return parser.parse_args()


def require_existing_path(raw: str | None, *, label: str) -> Path:
    if raw is None or str(raw).strip() == "":
        raise ValueError(f"{label} is required.")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def load_prompts(path: Path) -> list[PromptSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts: list[PromptSpec] = []
    for item in data:
        name = str(item["name"]).strip()
        prompt = str(item["prompt"]).strip()
        if not name or not prompt:
            raise ValueError(f"Invalid prompt entry in {path}: {item}")
        prompts.append(PromptSpec(name=name, prompt=prompt))
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def list_images(input_dir: Path, num_shards: int, shard_index: int, limit: int | None) -> list[Path]:
    images = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    images = [p for idx, p in enumerate(images) if idx % num_shards == shard_index]
    if limit is not None:
        images = images[:limit]
    return images


def configure_pipeline_memory(pipe, device: str) -> None:
    if str(device).startswith("cuda"):
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    src = image.convert("RGB")
    scale = min(target_w / src.width, target_h / src.height)
    new_w = max(1, int(round(src.width * scale)))
    new_h = max(1, int(round(src.height * scale)))
    resized = src.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, color=(18, 18, 18))
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def draw_label(image: Image.Image, label: str) -> Image.Image:
    font = ImageFont.load_default()
    padding = 10
    probe = ImageDraw.Draw(image)
    text_bbox = probe.textbbox((0, 0), label, font=font)
    label_h = (text_bbox[3] - text_bbox[1]) + padding * 2
    canvas = Image.new("RGB", (image.width, image.height + label_h), color=(18, 18, 18))
    canvas.paste(image, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, padding), label, fill=(240, 240, 240), font=font)
    return canvas


def make_comparison_frame(input_image: Image.Image, output_image: Image.Image, label: str) -> Image.Image:
    target_size = output_image.size
    input_panel = fit_image(input_image, target_size)
    output_panel = fit_image(output_image, target_size)
    merged = Image.new("RGB", (target_size[0] * 2, target_size[1]), color=(0, 0, 0))
    merged.paste(input_panel, (0, 0))
    merged.paste(output_panel, (target_size[0], 0))
    return draw_label(merged, label)


def make_contact_sheet(
    input_image: Image.Image,
    frames: list[tuple[str, Image.Image]],
    cell_size: tuple[int, int],
) -> Image.Image:
    tiles: list[Image.Image] = [draw_label(fit_image(input_image, cell_size), "input")]
    for label, frame in frames:
        tiles.append(draw_label(fit_image(frame, cell_size), label))

    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(12, 12, 12))
    for idx, tile in enumerate(tiles):
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        sheet.paste(tile, (x, y))
    return sheet


def save_gif(frames: list[Image.Image], save_path: Path, duration_ms: int) -> None:
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def main() -> None:
    args = parse_args()
    from diffusers import ControlLightPipeline

    prompts = load_prompts(Path(args.prompts_file))
    alphas = [float(item.strip()) for item in args.alphas.split(",") if item.strip()]
    if not alphas:
        raise ValueError("No alpha values provided.")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_config = {
        "model_path": args.model_path,
        "lora_path": args.lora_path,
        "input_dir": str(input_dir),
        "prompts_file": str(Path(args.prompts_file)),
        "alphas": alphas,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "max_sequence_length": args.max_sequence_length,
        "torch_dtype": args.torch_dtype,
        "device": args.device,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "limit": args.limit,
    }
    (output_root / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (output_root / "prompts.json").write_text(
        json.dumps([prompt.__dict__ for prompt in prompts], indent=2),
        encoding="utf-8",
    )

    images = list_images(
        input_dir=input_dir,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        limit=args.limit,
    )
    print(f"Found {len(images)} images for shard {args.shard_index}/{args.num_shards}.")

    model_path = require_existing_path(args.model_path, label="Model path")
    lora_path = require_existing_path(args.lora_path, label="LoRA path")
    pipe = ControlLightPipeline.from_pretrained(
        str(model_path),
        torch_dtype=DTYPE_MAP[args.torch_dtype],
        default_lora_path=str(lora_path),
    )
    configure_pipeline_memory(pipe=pipe, device=args.device)

    summary: list[dict[str, object]] = []
    for image_index, image_path in enumerate(images, start=1):
        item_dir = output_root / image_path.stem
        gif_path = item_dir / "comparison.gif"
        metadata_path = item_dir / "metadata.json"
        if args.skip_existing and gif_path.exists() and metadata_path.exists():
            print(f"[{image_index}/{len(images)}] skip {image_path.name}")
            continue

        print(f"[{image_index}/{len(images)}] process {image_path.name}")
        item_dir.mkdir(parents=True, exist_ok=True)
        input_image = Image.open(image_path).convert("RGB")
        input_copy_path = item_dir / "input.png"
        input_image.save(input_copy_path)

        frames_for_gif: list[Image.Image] = []
        outputs_for_grid: list[tuple[str, Image.Image]] = []
        item_metadata = {
            "image": str(image_path),
            "input_copy": str(input_copy_path),
            "gif": str(gif_path),
            "grid": str(item_dir / "grid.jpg"),
            "outputs": [],
        }

        for prompt_spec in prompts:
            prompt_dir = item_dir / prompt_spec.name
            prompt_dir.mkdir(parents=True, exist_ok=True)
            for alpha in alphas:
                result = pipe(
                    image=input_image,
                    prompt=prompt_spec.prompt,
                    alpha=float(alpha),
                    seed=args.seed,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    max_sequence_length=args.max_sequence_length,
                )
                output_image = result.images[0].convert("RGB")
                alpha_tag = f"{alpha:.2f}"
                output_path = prompt_dir / f"alpha_{alpha_tag}.png"
                output_image.save(output_path)

                label = f"{prompt_spec.name} | alpha={alpha_tag}"
                frames_for_gif.append(make_comparison_frame(input_image, output_image, label))
                outputs_for_grid.append((label, output_image))
                item_metadata["outputs"].append(
                    {
                        "prompt_name": prompt_spec.name,
                        "prompt": prompt_spec.prompt,
                        "alpha": float(alpha),
                        "output": str(output_path),
                    }
                )

        grid_path = item_dir / "grid.jpg"
        grid_image = make_contact_sheet(input_image=input_image, frames=outputs_for_grid, cell_size=(384, 384))
        grid_image.save(grid_path, quality=95)
        save_gif(frames_for_gif, gif_path, duration_ms=args.gif_duration_ms)
        metadata_path.write_text(json.dumps(item_metadata, indent=2), encoding="utf-8")
        summary.append(
            {
                "image": str(image_path),
                "gif": str(gif_path),
                "grid": str(grid_path),
                "num_outputs": len(item_metadata["outputs"]),
            }
        )

    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved outputs under {output_root}")


if __name__ == "__main__":
    main()
