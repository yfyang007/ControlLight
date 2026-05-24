#!/usr/bin/env python3
"""Resume-aware watchdog for ControlLight multi-GPU training."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml


DEFAULT_PROCESS_REGEX = r"run\.py .*train_flux2klein.*\.ya?ml"
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_NAME = "controllight_lora_train"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] [train-watchdog] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resume-aware watchdog for ControlLight training.")
    p.add_argument("--workdir", default=str(DEFAULT_REPO_ROOT), help="Repository root.")
    p.add_argument(
        "--template-config",
        default=str(DEFAULT_REPO_ROOT / "config" / "train_flux2klein_lora.yaml"),
        help="Template YAML used to generate the resumed config.",
    )
    p.add_argument(
        "--generated-config",
        default="/tmp/controllight_train_resume_latest.yaml",
        help="Path for the generated resume config.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_REPO_ROOT / "output" / DEFAULT_RUN_NAME),
        help="Directory containing saved checkpoints.",
    )
    p.add_argument(
        "--checkpoint-prefix",
        default=DEFAULT_RUN_NAME,
        help="Checkpoint file prefix before _000000123.safetensors.",
    )
    p.add_argument("--process-regex", default=DEFAULT_PROCESS_REGEX, help="Regex used to detect active training processes.")
    p.add_argument("--gpus", default="0,1,2,3", help="Visible GPU list, for example 0,1,2,3.")
    p.add_argument("--num-processes", type=int, default=4, help="Accelerate process count.")
    p.add_argument("--port-base", type=int, default=30000, help="Base port for restarted jobs.")
    p.add_argument("--stop-step", type=int, default=6000, help="Stop watchdog once the latest checkpoint reaches this step.")
    p.add_argument("--save-every", type=int, default=500, help="Checkpoint interval to inject into the resume config.")
    p.add_argument("--poll-seconds", type=float, default=30.0, help="Polling interval while waiting for training to continue.")
    p.add_argument("--cooldown-seconds", type=float, default=20.0, help="Sleep time after a finished or failed restart attempt.")
    p.add_argument("--train-log", default=f"output/{DEFAULT_RUN_NAME}/train_4gpu.log", help="Train log file path, relative to workdir or absolute.")
    p.add_argument("--watchdog-log", default=f"output/{DEFAULT_RUN_NAME}/watchdog.log", help="Watchdog log file path, relative to workdir or absolute.")
    return p.parse_args()


def append_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def checkpoint_step(path: Path, prefix: str) -> int | None:
    m = re.fullmatch(rf"{re.escape(prefix)}_(\d{{9}})\.safetensors", path.name)
    return int(m.group(1)) if m else None


def latest_checkpoint(checkpoint_dir: Path, prefix: str) -> tuple[int | None, Path | None]:
    best: tuple[int | None, Path | None] = (None, None)
    for path in checkpoint_dir.glob(f"{prefix}_*.safetensors"):
        step = checkpoint_step(path, prefix)
        if step is not None and (best[0] is None or step > best[0]):
            best = (step, path)
    return best


def ps_lines() -> list[str]:
    out = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,stat=,etime=,cmd="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    ).stdout
    return out.splitlines()


def matching_training_processes(pattern: re.Pattern[str], self_pid: int) -> list[str]:
    matches = []
    for line in ps_lines():
        if str(self_pid) in line and "watchdog" in line:
            continue
        if pattern.search(line):
            if "monitor_flux2klein_alpha_vis" in line or "visualize_flux2klein" in line:
                continue
            matches.append(line)
    return matches


def write_resume_config(template: Path, out_path: Path, start_step: int | None, save_every: int) -> None:
    cfg = yaml.safe_load(template.read_text())
    proc = cfg["config"]["process"][0]
    proc["train"]["start_step"] = 0 if start_step is None else start_step
    proc["save"]["save_every"] = save_every
    alpha_vis = proc.get("alpha_visualization")
    if isinstance(alpha_vis, dict):
        alpha_vis["enabled"] = False
        alpha_vis["before_first_step"] = False
        alpha_vis["every"] = 0
        alpha_vis["distributed_periodic"] = False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


def build_train_command(args: argparse.Namespace, config_path: Path, restart_idx: int) -> str:
    port = args.port_base + (restart_idx % 100)
    run_id = f"controllight_watchdog_{int(time.time())}_{restart_idx}"
    repo_root = Path(args.workdir).resolve()
    return (
        f"export CUDA_VISIBLE_DEVICES={args.gpus}; "
        "export HF_HUB_ENABLE_HF_TRANSFER=1; "
        "export NO_ALBUMENTATIONS_UPDATE=1; "
        "export AI_TOOLKIT_FS_BARRIER_TIMEOUT_SEC=7200; "
        f"export PYTHONPATH={repo_root}:$PYTHONPATH; "
        f"export AI_TOOLKIT_RUN_ID={run_id}; "
        f"CONFIG={config_path} GPUS={args.gpus} NUM_PROCESSES={args.num_processes} MAIN_PROCESS_PORT={port} "
        f"bash {repo_root}/scripts/train/train_multigpu.sh "
        f"2>&1 | tee -a {args.train_log}"
    )


def resolve_log_path(workdir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (workdir / path)


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    template = Path(args.template_config).resolve()
    generated = Path(args.generated_config)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    train_log = resolve_log_path(workdir, args.train_log)
    watchdog_log = resolve_log_path(workdir, args.watchdog_log)
    process_re = re.compile(args.process_regex)

    restart_idx = 0
    log(f"watching workdir={workdir}")
    log(f"process_regex={args.process_regex}")
    log(f"checkpoint_dir={checkpoint_dir} stop_step={args.stop_step} save_every={args.save_every}")
    append_file(watchdog_log, f"\n[{now()}] watchdog started\n")

    while True:
        latest_step, latest_path = latest_checkpoint(checkpoint_dir, args.checkpoint_prefix)
        if latest_step is not None and latest_step >= args.stop_step:
            msg = f"latest checkpoint step {latest_step} >= stop_step {args.stop_step}; not restarting"
            log(msg)
            append_file(watchdog_log, f"[{now()}] {msg}\n")
            return 0

        matches = matching_training_processes(process_re, os.getpid())
        if matches:
            time.sleep(args.poll_seconds)
            continue

        latest_step, latest_path = latest_checkpoint(checkpoint_dir, args.checkpoint_prefix)
        start_step = latest_step
        msg = f"no training process found; latest checkpoint={latest_path} step={latest_step}; restarting"
        log(msg)
        append_file(watchdog_log, f"[{now()}] {msg}\n")
        write_resume_config(template, generated, start_step, args.save_every)
        cmd = build_train_command(args, generated, restart_idx)
        restart_idx += 1
        append_file(watchdog_log, f"[{now()}] command: {cmd}\n")
        proc = subprocess.Popen(["/bin/bash", "-lc", cmd], cwd=str(workdir))
        rc = proc.wait()
        latest_step_after, _ = latest_checkpoint(checkpoint_dir, args.checkpoint_prefix)
        if (
            rc == 0
            and start_step is not None
            and latest_step_after == start_step
            and start_step + args.save_every >= args.stop_step
        ):
            msg = (
                "training exited cleanly near stop_step without creating a newer checkpoint; "
                "assuming training completed and stopping watchdog"
            )
            log(msg)
            append_file(watchdog_log, f"[{now()}] {msg}\n")
            return 0
        msg = f"training command exited rc={rc}; cooldown {args.cooldown_seconds}s"
        log(msg)
        append_file(watchdog_log, f"[{now()}] {msg}\n")
        time.sleep(args.cooldown_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
