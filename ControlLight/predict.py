from __future__ import annotations

"""Public-facing prediction CLI for ControlLight."""

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

from ControlLight.bootstrap import bootstrap_local_paths

_REPO_ROOT = bootstrap_local_paths()

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from toolkit.buckets import get_bucket_for_image_size


DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

DEFAULT_MODEL_PATH = None
DEFAULT_LORA_PATH = str((_REPO_ROOT / "weights" / "controllight.safetensors").resolve())
DEFAULT_PROMPT = (
    "Enhance this low-light image by lifting exposure and recovering visible details "
    "while preserving identity, geometry, atmosphere, natural colors, and avoiding halos, "
    "noise, over-sharpening, or overexposure."
)
DEFAULT_IMAGE_ALPHA = 0.54
DEFAULT_BATCH_ALPHAS = [0.25, 0.50, 0.75, 1.00]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_alphas(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("No alpha values were parsed from --alphas.")
    return values


def require_existing_path(raw: str | None, *, label: str) -> str:
    if raw is None or str(raw).strip() == "":
        raise ValueError(
            f"{label} is required. Pass it explicitly or configure your shell before running."
        )
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return str(path)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cap_pixels(image: Image.Image, limit_pixels: int = 1024**2) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    pixel_count = image.width * image.height
    if pixel_count <= limit_pixels:
        return image
    scale = (limit_pixels / pixel_count) ** 0.5
    new_w = max(1, int(image.width * scale))
    new_h = max(1, int(image.height * scale))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    src = image.convert("RGB")
    target_w, target_h = size
    scale = min(target_w / src.width, target_h / src.height)
    new_w = max(1, int(round(src.width * scale)))
    new_h = max(1, int(round(src.height * scale)))
    resized = src.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, color=(10, 16, 24))
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def draw_label(image: Image.Image, label: str) -> Image.Image:
    font = ImageFont.load_default()
    padding = 12
    probe = ImageDraw.Draw(image)
    left, top, right, bottom = probe.textbbox((0, 0), label, font=font)
    label_h = (bottom - top) + padding * 2
    canvas = Image.new("RGB", (image.width, image.height + label_h), color=(10, 16, 24))
    canvas.paste(image, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, padding), label, fill=(242, 247, 250), font=font)
    return canvas


def save_gif(frames: list[Image.Image], path: Path, duration_ms: int) -> None:
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def make_grid(
    input_image: Image.Image,
    labeled_outputs: list[tuple[str, Image.Image]],
    cell_size: tuple[int, int] = (256, 256),
) -> Image.Image:
    tiles = [draw_label(fit_image(input_image, cell_size), "input")]
    for label, image in labeled_outputs:
        tiles.append(draw_label(fit_image(image, cell_size), label))
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    grid = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(8, 14, 22))
    for idx, tile in enumerate(tiles):
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        grid.paste(tile, (x, y))
    return grid


def make_comparison_frame(input_image: Image.Image, output_image: Image.Image, label: str) -> Image.Image:
    target_size = output_image.size
    left = fit_image(input_image, target_size)
    right = fit_image(output_image, target_size)
    merged = Image.new("RGB", (target_size[0] * 2, target_size[1]), color=(0, 0, 0))
    merged.paste(left, (0, 0))
    merged.paste(right, (target_size[0], 0))
    return draw_label(merged, f"input | {label}")


def bucket_resize_center_crop(image: Image.Image, resolution: int, divisibility: int) -> tuple[Image.Image, dict]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    bucket = get_bucket_for_image_size(
        width=image.width,
        height=image.height,
        resolution=resolution,
        divisibility=divisibility,
    )
    if bucket["width"] > bucket["height"]:
        scaled_h = bucket["height"]
        scaled_w = int(image.width * (bucket["height"] / image.height))
    else:
        scaled_w = bucket["width"]
        scaled_h = int(image.height * (bucket["width"] / image.width))
    resized = image.resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)
    left = max(0, (resized.width - bucket["width"]) // 2)
    top = max(0, (resized.height - bucket["height"]) // 2)
    cropped = resized.crop((left, top, left + bucket["width"], top + bucket["height"]))
    meta = {
        "original_size": [image.width, image.height],
        "bucket_size": [bucket["width"], bucket["height"]],
        "scaled_size": [scaled_w, scaled_h],
        "crop_box": [left, top, left + bucket["width"], top + bucket["height"]],
        "resolution": resolution,
        "divisibility": divisibility,
    }
    return cropped, meta


def make_shared_latents_4d(
    pipe,
    *,
    batch_size: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    multiple_of = pipe.vae_scale_factor * 2
    height = 2 * (int(height) // multiple_of)
    width = 2 * (int(width) // multiple_of)
    num_channels_latents = pipe.transformer.config.in_channels // 4
    shape = (batch_size, num_channels_latents * 4, height // 2, width // 2)
    generator_device = device if str(device).startswith("cuda") else torch.device("cpu")
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    return latents.detach().clone()


def configure_pipeline_memory(pipe, device: str) -> None:
    if str(device).startswith("cuda"):
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)


def build_pipeline(args):
    from diffusers import ControlLightPipeline

    pipe = ControlLightPipeline.from_pretrained(
        require_existing_path(args.model_path, label="Model path"),
        torch_dtype=DTYPE_MAP[args.torch_dtype],
        default_lora_path=require_existing_path(args.lora_path, label="LoRA path"),
        default_prompt=args.prompt,
        default_num_inference_steps=args.num_inference_steps,
        default_guidance_scale=args.guidance_scale,
        default_max_sequence_length=args.max_sequence_length,
    )
    configure_pipeline_memory(pipe=pipe, device=args.device)
    return pipe


def run_predict_image(args) -> None:
    pipe = build_pipeline(args)
    image = Image.open(args.input).convert("RGB")
    result = pipe(
        image=image,
        prompt=args.prompt,
        alpha=args.alpha,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output_path)
    print(f"Saved output image to {output_path}")


def collect_batch_items(input_root: Path, requested_slugs: set[str]) -> tuple[list[tuple[str, Path]], str]:
    if not input_root.is_dir():
        raise ValueError("--input must be a directory in batch mode.")
    sources_dir = input_root / "sources" if (input_root / "sources").is_dir() else input_root
    items: list[tuple[str, Path]] = []
    for path in sorted(sources_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            if requested_slugs and path.stem not in requested_slugs:
                continue
            items.append((path.stem, path))
    return items, ("design_materials" if sources_dir != input_root else "directory")


def run_predict_batch(args) -> None:
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    requested_slugs = {slug.strip() for slug in args.slugs.split(",") if slug.strip()}
    alphas = parse_alphas(args.alphas)
    items, input_mode = collect_batch_items(input_root, requested_slugs)
    if not items:
        raise ValueError("No input images found.")

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sources_dir = output_root / "sources"
    results_dir = output_root / "results"
    sources_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline(args)
    manifest = {
        "mode": "batch",
        "input_root": str(input_root),
        "input_mode": input_mode,
        "output_root": str(output_root),
        "model_path": args.model_path,
        "lora_path": args.lora_path,
        "prompt": args.prompt,
        "alphas": alphas,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "max_sequence_length": args.max_sequence_length,
        "resolution": args.resolution,
        "bucket_divisibility": args.bucket_divisibility,
        "gif_duration_ms": args.gif_duration_ms,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "compat_mode": "watchdog",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "items": [],
    }
    write_json(output_root / "manifest.json", manifest)

    for index, (slug, src_path) in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {slug}", flush=True)
        item_dir = results_dir / slug
        outputs_dir = item_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        original = Image.open(src_path).convert("RGB")
        prepared = cap_pixels(original)
        work_image, work_meta = bucket_resize_center_crop(original, args.resolution, args.bucket_divisibility)
        source_copy = sources_dir / f"{slug}.png"
        input_copy = item_dir / "input.jpg"
        original.save(source_copy)
        prepared.save(input_copy, quality=95)

        shared_latents = make_shared_latents_4d(
            pipe,
            batch_size=1,
            height=work_image.height,
            width=work_image.width,
            dtype=pipe.transformer.dtype,
            device=pipe._execution_device,
            seed=args.seed,
        )

        labeled_outputs: list[tuple[str, Image.Image]] = []
        gif_frames: list[Image.Image] = [make_comparison_frame(work_image, work_image, "input")]
        output_records: list[dict] = []

        for alpha in alphas:
            started = time.perf_counter()
            result = pipe(
                image=work_image,
                prompt=args.prompt,
                alpha=alpha,
                seed=None,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                max_sequence_length=args.max_sequence_length,
                watchdog_compat=True,
                latents=shared_latents.clone(),
            )
            output_image = result.images[0].convert("RGB")
            elapsed = time.perf_counter() - started
            label = f"alpha_{alpha:.2f}"
            output_path = outputs_dir / f"{label}.png"
            output_image.save(output_path)
            labeled_outputs.append((f"Alpha {alpha:.2f}", output_image))
            gif_frames.append(make_comparison_frame(work_image, output_image, label))
            output_records.append(
                {
                    "alpha": alpha,
                    "image": str(output_path),
                    "seconds": elapsed,
                }
            )
            del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        grid_path = item_dir / "grid.jpg"
        gif_path = item_dir / "comparison.gif"
        make_grid(work_image, labeled_outputs).save(grid_path, quality=95)
        save_gif(gif_frames, gif_path, args.gif_duration_ms)

        metadata = {
            "slug": slug,
            "source_path": str(src_path),
            "source_copy": str(source_copy),
            "input_copy": str(input_copy),
            "prompt": args.prompt,
            "alphas": alphas,
            "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "max_sequence_length": args.max_sequence_length,
            "compat_mode": "watchdog",
            "work_meta": work_meta,
            "grid_path": str(grid_path),
            "gif_path": str(gif_path),
            "outputs": output_records,
        }
        metadata_path = item_dir / "metadata.json"
        write_json(metadata_path, metadata)
        manifest["items"].append(
            {
                "slug": slug,
                "metadata_path": str(metadata_path),
                "outputs": output_records,
            }
        )
        write_json(output_root / "manifest.json", manifest)

        del original
        del prepared
        del work_image
        del shared_latents
        del labeled_outputs
        del gif_frames
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Saved batch prediction outputs under {output_root}")


def run_predict_strengths(args) -> None:
    run_predict_batch(args)


def run_predict_four(args) -> None:
    args.alphas = ",".join(f"{alpha:.2f}" for alpha in DEFAULT_BATCH_ALPHAS)
    run_predict_batch(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ControlLight prediction CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predict.py predict-image --input input.jpg --output output.png --model-path ./models/FLUX.2-klein-base-9B --device cuda\n"
            "  python predict.py predict-strengths --input ./assets --output ./runs/assets_strengths --model-path ./models/FLUX.2-klein-base-9B --alphas 0.20,0.40,0.60,0.80\n"
            "  python predict.py predict-four --input ./design_materials --output ./runs/design_materials_four --model-path ./models/FLUX.2-klein-base-9B"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add_shared_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Base FLUX.2-Klein model path.")
        subparser.add_argument("--lora-path", default=DEFAULT_LORA_PATH, help="ControlLight LoRA checkpoint.")
        subparser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Restoration prompt.")
        subparser.add_argument("--seed", type=int, default=42, help="Random seed.")
        subparser.add_argument("--num-inference-steps", type=int, default=20, help="Number of denoising steps.")
        subparser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale.")
        subparser.add_argument("--max-sequence-length", type=int, default=512, help="Maximum token length.")
        subparser.add_argument("--device", default="cuda", help="Execution device, for example cuda or cpu.")
        subparser.add_argument(
            "--torch-dtype",
            default="bfloat16",
            choices=sorted(DTYPE_MAP),
            help="Inference dtype.",
        )

    image_parser = subparsers.add_parser(
        "predict-image",
        help="Enhance one input image.",
        description="Run ControlLight on a single image.",
    )
    add_shared_args(image_parser)
    image_parser.add_argument("--input", required=True, help="Input image path.")
    image_parser.add_argument("--output", required=True, help="Output image path.")
    image_parser.add_argument("--alpha", type=float, default=DEFAULT_IMAGE_ALPHA, help="LoRA strength.")
    image_parser.set_defaults(handler=run_predict_image)

    strengths_parser = subparsers.add_parser(
        "predict-strengths",
        help="Enhance one directory across a custom alpha sweep.",
        description="Run a custom multi-strength alpha sweep over a directory or design_materials source root.",
    )
    add_shared_args(strengths_parser)
    strengths_parser.add_argument("--input", required=True, help="Input directory or design_materials root.")
    strengths_parser.add_argument("--output", required=True, help="Output directory.")
    strengths_parser.add_argument(
        "--alphas",
        default=",".join(f"{alpha:.2f}" for alpha in DEFAULT_BATCH_ALPHAS),
        help="Comma-separated alpha values, for example 0.25,0.50,0.75,1.00.",
    )
    strengths_parser.add_argument("--slugs", default="", help="Optional comma-separated item stems to keep.")
    strengths_parser.add_argument("--resolution", type=int, default=1024, help="Bucket resolution.")
    strengths_parser.add_argument("--bucket-divisibility", type=int, default=16, help="Bucket divisibility.")
    strengths_parser.add_argument("--gif-duration-ms", type=int, default=450, help="GIF frame duration in milliseconds.")
    strengths_parser.add_argument("--overwrite", action="store_true", help="Delete the output directory before running.")
    strengths_parser.set_defaults(handler=run_predict_strengths)

    four_parser = subparsers.add_parser(
        "predict-four",
        help="Enhance one directory with the fixed public four-strength sweep.",
        description="Run the fixed four-strength alpha sweep 0.25, 0.50, 0.75, 1.00 over a directory or design_materials source root.",
    )
    add_shared_args(four_parser)
    four_parser.add_argument("--input", required=True, help="Input directory or design_materials root.")
    four_parser.add_argument("--output", required=True, help="Output directory.")
    four_parser.add_argument("--slugs", default="", help="Optional comma-separated item stems to keep.")
    four_parser.add_argument("--resolution", type=int, default=1024, help="Bucket resolution.")
    four_parser.add_argument("--bucket-divisibility", type=int, default=16, help="Bucket divisibility.")
    four_parser.add_argument("--gif-duration-ms", type=int, default=450, help="GIF frame duration in milliseconds.")
    four_parser.add_argument("--overwrite", action="store_true", help="Delete the output directory before running.")
    four_parser.set_defaults(handler=run_predict_four)

    batch_parser = subparsers.add_parser(
        "predict-batch",
        help="Deprecated alias for predict-four.",
        description="Deprecated alias for predict-four. Runs the fixed four-strength sweep 0.25, 0.50, 0.75, 1.00.",
    )
    add_shared_args(batch_parser)
    batch_parser.add_argument("--input", required=True, help="Input directory or design_materials root.")
    batch_parser.add_argument("--output", required=True, help="Output directory.")
    batch_parser.add_argument("--slugs", default="", help="Optional comma-separated item stems to keep.")
    batch_parser.add_argument("--resolution", type=int, default=1024, help="Bucket resolution.")
    batch_parser.add_argument("--bucket-divisibility", type=int, default=16, help="Bucket divisibility.")
    batch_parser.add_argument("--gif-duration-ms", type=int, default=450, help="GIF frame duration in milliseconds.")
    batch_parser.add_argument("--overwrite", action="store_true", help="Delete the output directory before running.")
    batch_parser.set_defaults(handler=run_predict_four)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
