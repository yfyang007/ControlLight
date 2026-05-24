from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ControlLight.bootstrap import bootstrap_local_paths

_REPO_ROOT = bootstrap_local_paths()

import cgi
import mimetypes

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
DEFAULT_MODEL_PATH = None
DEFAULT_LORA_PATH = str((_REPO_ROOT / "weights" / "controllight.safetensors").resolve())
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "outputs" / "web_demo"
DEFAULT_STATIC_DIR = _REPO_ROOT / "web_demo"
DEFAULT_ALPHAS = [0.25, 0.50, 0.75, 1.00]
MAX_UPLOAD_MB = 30


@dataclass(frozen=True)
class DemoConfig:
    model_path: str
    lora_path: str
    output_root: Path
    static_dir: Path
    host: str
    port: int
    device: str
    torch_dtype: str
    default_prompt: str | None
    alphas: list[float]
    seed: int
    num_inference_steps: int
    guidance_scale: float
    max_sequence_length: int
    gif_duration_ms: int


class PipelineManager:
    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self._pipe = None
        self._lock = threading.Lock()

    def _configure_pipeline_memory(self, pipe) -> None:
        if str(self.config.device).startswith("cuda"):
            pipe.enable_model_cpu_offload(device=self.config.device)
        else:
            pipe.to(self.config.device)

    def get_pipe(self):
        if self._pipe is None:
            with self._lock:
                if self._pipe is None:
                    from diffusers import ControlLightPipeline

                    pipe = ControlLightPipeline.from_pretrained(
                        self.config.model_path,
                        torch_dtype=DTYPE_MAP[self.config.torch_dtype],
                        default_lora_path=self.config.lora_path,
                    )
                    self._configure_pipeline_memory(pipe)
                    self._pipe = pipe
        return self._pipe

    def run_inference(
        self,
        image: Image.Image,
        prompt: str | None,
        alpha: float,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
        max_sequence_length: int,
    ) -> Image.Image:
        pipe = self.get_pipe()
        with self._lock:
            result = pipe(
                image=image,
                prompt=prompt,
                alpha=float(alpha),
                seed=seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
            )
        return result.images[0].convert("RGB")


class DemoApp:
    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self.pipeline_manager = PipelineManager(config=config)
        self.output_root = config.output_root.resolve()
        self.static_dir = config.static_dir.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "runs").mkdir(parents=True, exist_ok=True)

    def build_handler(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "ControlLightDemo/0.1"

            def log_message(self, format: str, *args: Any) -> None:
                sys.stdout.write("[%s] %s\n" % (self.log_date_time_string(), format % args))

            def do_GET(self) -> None:
                try:
                    parsed = urlparse(self.path)
                    route = parsed.path
                    if route in {"/", "/index.html"}:
                        return self._serve_file(app.static_dir / "index.html", cache=False)
                    if route.startswith("/static/"):
                        rel = route.removeprefix("/static/")
                        return self._serve_static(rel)
                    if route.startswith("/runs/"):
                        rel = route.removeprefix("/")
                        return self._serve_output(rel)
                    if route == "/api/config":
                        return self._send_json(
                            {
                                "title": "ControlLight Local Demo",
                                "default_prompt": app.config.default_prompt or "",
                                "alphas": app.config.alphas,
                                "seed": app.config.seed,
                                "num_inference_steps": app.config.num_inference_steps,
                                "guidance_scale": app.config.guidance_scale,
                                "max_sequence_length": app.config.max_sequence_length,
                                "gif_duration_ms": app.config.gif_duration_ms,
                                "device": app.config.device,
                                "torch_dtype": app.config.torch_dtype,
                                "model_path": app.config.model_path,
                                "lora_path": app.config.lora_path,
                            }
                        )
                    return self._send_error(HTTPStatus.NOT_FOUND, "Route not found")
                except Exception as exc:  # pragma: no cover - defensive branch
                    traceback.print_exc()
                    return self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"GET failed: {exc}")

            def do_POST(self) -> None:
                try:
                    parsed = urlparse(self.path)
                    if parsed.path == "/api/infer":
                        return self._handle_infer()
                    return self._send_error(HTTPStatus.NOT_FOUND, "Route not found")
                except Exception as exc:  # pragma: no cover - defensive branch
                    traceback.print_exc()
                    return self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"POST failed: {exc}")

            def _handle_infer(self) -> None:
                form = self._parse_multipart_form()
                if form is None:
                    return

                upload_item = form["image"] if "image" in form else None
                if upload_item is None or not getattr(upload_item, "filename", None):
                    return self._send_error(HTTPStatus.BAD_REQUEST, "Missing uploaded image file.")

                raw_filename = Path(upload_item.filename).name
                suffix = Path(raw_filename).suffix.lower() or ".png"
                if suffix not in IMAGE_EXTS:
                    return self._send_error(
                        HTTPStatus.BAD_REQUEST,
                        f"Unsupported file type: {suffix}. Allowed: {sorted(IMAGE_EXTS)}",
                    )

                content = upload_item.file.read()
                if not content:
                    return self._send_error(HTTPStatus.BAD_REQUEST, "Uploaded image is empty.")
                if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
                    return self._send_error(
                        HTTPStatus.BAD_REQUEST,
                        f"Uploaded file is larger than {MAX_UPLOAD_MB} MB.",
                    )

                prompt = (form.getfirst("prompt") or app.config.default_prompt or "").strip()
                if prompt == "":
                    prompt = None

                try:
                    alphas = parse_alpha_string(form.getfirst("alphas"), default=app.config.alphas)
                    seed = parse_int(form.getfirst("seed"), default=app.config.seed)
                    num_inference_steps = parse_int(
                        form.getfirst("num_inference_steps"),
                        default=app.config.num_inference_steps,
                    )
                    guidance_scale = parse_float(
                        form.getfirst("guidance_scale"),
                        default=app.config.guidance_scale,
                    )
                    max_sequence_length = parse_int(
                        form.getfirst("max_sequence_length"),
                        default=app.config.max_sequence_length,
                    )
                    gif_duration_ms = parse_int(
                        form.getfirst("gif_duration_ms"),
                        default=app.config.gif_duration_ms,
                    )
                except ValueError as exc:
                    return self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

                try:
                    source_image = Image.open(io_from_bytes(content)).convert("RGB")
                except Exception as exc:
                    return self._send_error(HTTPStatus.BAD_REQUEST, f"Failed to decode image: {exc}")

                run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}-{threading.get_ident()}"
                run_dir = app.output_root / "runs" / run_id
                uploads_dir = run_dir / "uploads"
                outputs_dir = run_dir / "outputs"
                uploads_dir.mkdir(parents=True, exist_ok=True)
                outputs_dir.mkdir(parents=True, exist_ok=True)

                safe_stem = slugify(Path(raw_filename).stem) or "upload"
                input_path = uploads_dir / f"{safe_stem}{suffix}"
                input_path.write_bytes(content)

                output_items: list[dict[str, Any]] = []
                gif_frames: list[Image.Image] = []
                grid_frames: list[tuple[str, Image.Image]] = []
                for alpha in alphas:
                    output_image = app.pipeline_manager.run_inference(
                        image=source_image,
                        prompt=prompt,
                        alpha=alpha,
                        seed=seed,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        max_sequence_length=max_sequence_length,
                    )
                    alpha_tag = format_alpha(alpha)
                    output_path = outputs_dir / f"alpha_{alpha_tag}.png"
                    output_image.save(output_path)
                    gif_frames.append(make_comparison_frame(source_image, output_image, f"alpha={alpha_tag}"))
                    grid_frames.append((f"Alpha {alpha_tag}", output_image))
                    output_items.append(
                        {
                            "alpha": alpha,
                            "label": f"Alpha {alpha_tag}",
                            "image_url": f"/runs/{run_id}/outputs/{output_path.name}",
                            "image_path": str(output_path),
                        }
                    )

                gif_path = run_dir / "comparison.gif"
                save_gif(gif_frames, gif_path, duration_ms=gif_duration_ms)

                grid_path = run_dir / "four_strengths_grid.jpg"
                grid_image = make_contact_sheet(
                    input_image=source_image,
                    frames=grid_frames,
                    cell_size=(384, 384),
                )
                grid_image.save(grid_path, quality=95)

                metadata = {
                    "run_id": run_id,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "input_file": raw_filename,
                    "input_path": str(input_path),
                    "prompt": prompt,
                    "alphas": alphas,
                    "seed": seed,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "max_sequence_length": max_sequence_length,
                    "gif_duration_ms": gif_duration_ms,
                    "gif_path": str(gif_path),
                    "grid_path": str(grid_path),
                    "outputs": output_items,
                }
                metadata_path = run_dir / "metadata.json"
                metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

                return self._send_json(
                    {
                        "ok": True,
                        "message": "Inference completed.",
                        "run_id": run_id,
                        "prompt": prompt or "",
                        "input_url": f"/runs/{run_id}/uploads/{input_path.name}",
                        "gif_url": f"/runs/{run_id}/{gif_path.name}",
                        "grid_url": f"/runs/{run_id}/{grid_path.name}",
                        "metadata_url": f"/runs/{run_id}/{metadata_path.name}",
                        "output_dir": str(run_dir),
                        "outputs": output_items,
                    },
                    status=HTTPStatus.OK,
                )

            def _parse_multipart_form(self):
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    self._send_error(HTTPStatus.BAD_REQUEST, "Content-Type must be multipart/form-data")
                    return None
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": content_type,
                        "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                    },
                )
                return form

            def _serve_output(self, relative_path: str) -> None:
                candidate = (app.output_root / relative_path).resolve()
                if not str(candidate).startswith(str(app.output_root)):
                    return self._send_error(HTTPStatus.FORBIDDEN, "Forbidden path")
                return self._serve_file(candidate, cache=False)

            def _serve_static(self, relative_path: str) -> None:
                candidate = (app.static_dir / relative_path).resolve()
                if not str(candidate).startswith(str(app.static_dir)):
                    return self._send_error(HTTPStatus.FORBIDDEN, "Forbidden path")
                return self._serve_file(candidate, cache=True)

            def _serve_file(self, path: Path, cache: bool) -> None:
                path = path.resolve()
                if not path.exists() or not path.is_file():
                    return self._send_error(HTTPStatus.NOT_FOUND, f"File not found: {path.name}")
                content_type, _ = mimetypes.guess_type(str(path))
                if content_type is None:
                    content_type = "application/octet-stream"
                data = path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
                self.end_headers()
                self.wfile.write(data)

            def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def _send_error(self, status: HTTPStatus, message: str) -> None:
                payload = {"ok": False, "error": message}
                self._send_json(payload, status=status)

        return Handler

    def serve(self) -> None:
        handler = self.build_handler()
        server = ThreadingHTTPServer((self.config.host, self.config.port), handler)
        print("=" * 72)
        print("ControlLight 本地 Demo 已启动")
        print(f"地址: http://{self.config.host}:{self.config.port}")
        print(f"模型: {self.config.model_path}")
        print(f"LoRA: {self.config.lora_path}")
        print(f"输出目录: {self.output_root}")
        print("提示: 首次推理会先加载模型，时间会明显更长。")
        print("=" * 72)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n收到 Ctrl+C，正在退出。")
        finally:
            server.server_close()


def io_from_bytes(content: bytes):
    from io import BytesIO

    return BytesIO(content)


def parse_alpha_string(raw: str | None, default: list[float]) -> list[float]:
    if raw is None or not raw.strip():
        return list(default)
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError as exc:
            raise ValueError(f"Invalid alpha value: {part}") from exc
    if not values:
        raise ValueError("No valid alpha values provided.")
    return values


def parse_int(raw: str | None, default: int) -> int:
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def parse_float(raw: str | None, default: float) -> float:
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def format_alpha(alpha: float) -> str:
    return f"{float(alpha):.2f}"


def slugify(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "-", value.strip())
    return value.strip("-._")


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    src = image.convert("RGB")
    scale = min(target_w / src.width, target_h / src.height)
    new_w = max(1, int(round(src.width * scale)))
    new_h = max(1, int(round(src.height * scale)))
    resized = src.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, color=(10, 16, 24))
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    canvas.paste(resized, offset)
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


def make_comparison_frame(input_image: Image.Image, output_image: Image.Image, label: str) -> Image.Image:
    target_size = output_image.size
    input_panel = fit_image(input_image, target_size)
    output_panel = fit_image(output_image, target_size)
    merged = Image.new("RGB", (target_size[0] * 2, target_size[1]), color=(0, 0, 0))
    merged.paste(input_panel, (0, 0))
    merged.paste(output_panel, (target_size[0], 0))
    return draw_label(merged, f"input  |  {label}")


def make_contact_sheet(
    input_image: Image.Image,
    frames: list[tuple[str, Image.Image]],
    cell_size: tuple[int, int],
) -> Image.Image:
    tiles: list[Image.Image] = [draw_label(fit_image(input_image, cell_size), "Input")]
    for label, frame in frames:
        tiles.append(draw_label(fit_image(frame, cell_size), label))

    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(8, 14, 22))
    for idx, tile in enumerate(tiles):
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        sheet.paste(tile, (x, y))
    return sheet


def save_gif(frames: list[Image.Image], save_path: Path, duration_ms: int) -> None:
    if not frames:
        raise ValueError("No frames available for GIF export.")
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local ControlLight upload demo.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=7860, help="Bind port.")
    parser.add_argument("--model-path", "--model_path", dest="model_path", default=DEFAULT_MODEL_PATH, help="Base model path. Required.")
    parser.add_argument("--lora-path", "--lora_path", dest="lora_path", default=DEFAULT_LORA_PATH, help="LoRA weights path.")
    parser.add_argument("--output-root", "--output_root", dest="output_root", default=str(DEFAULT_OUTPUT_ROOT), help="Where runs are written.")
    parser.add_argument("--static-dir", "--static_dir", dest="static_dir", default=str(DEFAULT_STATIC_DIR), help="Static frontend directory.")
    parser.add_argument("--default-prompt", "--default_prompt", dest="default_prompt", default="", help="Optional default prompt used when prompt input is empty.")
    parser.add_argument("--alphas", default=",".join(str(x) for x in DEFAULT_ALPHAS), help="Comma-separated alpha values.")
    parser.add_argument("--seed", type=int, default=42, help="Seed reused across the four strengths.")
    parser.add_argument("--num-inference-steps", "--num_inference_steps", dest="num_inference_steps", type=int, default=20, help="Number of denoising steps.")
    parser.add_argument("--guidance-scale", "--guidance_scale", dest="guidance_scale", type=float, default=1.0, help="Guidance scale.")
    parser.add_argument("--max-sequence-length", "--max_sequence_length", dest="max_sequence_length", type=int, default=512, help="Prompt max sequence length.")
    parser.add_argument("--gif-duration-ms", "--gif_duration_ms", dest="gif_duration_ms", type=int, default=500, help="Frame duration in milliseconds.")
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


def validate_args(args: argparse.Namespace) -> DemoConfig:
    if args.model_path is None or str(args.model_path).strip() == "":
        raise ValueError("Model path is required. Pass --model-path.")
    model_path = Path(args.model_path).expanduser().resolve()
    lora_path = Path(args.lora_path).expanduser().resolve()
    static_dir = Path(args.static_dir)
    output_root = Path(args.output_root)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA path does not exist: {lora_path}")
    if not static_dir.exists():
        raise FileNotFoundError(f"Static directory does not exist: {static_dir}")
    alphas = parse_alpha_string(args.alphas, default=DEFAULT_ALPHAS)
    return DemoConfig(
        model_path=str(model_path),
        lora_path=str(lora_path),
        output_root=output_root,
        static_dir=static_dir,
        host=args.host,
        port=int(args.port),
        device=args.device,
        torch_dtype=args.torch_dtype,
        default_prompt=args.default_prompt or None,
        alphas=alphas,
        seed=int(args.seed),
        num_inference_steps=int(args.num_inference_steps),
        guidance_scale=float(args.guidance_scale),
        max_sequence_length=int(args.max_sequence_length),
        gif_duration_ms=int(args.gif_duration_ms),
    )


def main() -> None:
    args = parse_args()
    config = validate_args(args)
    app = DemoApp(config=config)
    app.serve()


if __name__ == "__main__":
    main()
