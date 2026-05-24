# ControlLight

ControlLight is a FLUX.2-Klein LoRA project for low-light enhancement.

This repository contains:

- public inference entrypoints
- public training entrypoints
- vendored local `diffusers/` integration required by ControlLight
- vendored training stack (`run.py`, `jobs/`, `toolkit/`, `extensions/`)

It does not bundle large runtime assets such as datasets, base models, or output results.

## Environment

Use one environment for both inference and training:

- `aitoolkit`

The shared bootstrap script is:

- `scripts/project_env.sh`

The shell wrappers activate `aitoolkit` automatically.

## Installation

See also: [`docs/INSTALL.md`](docs/INSTALL.md)

```bash
python -m pip install --upgrade pip
python -m pip install -e diffusers
python -m pip install -r requirements.txt
python -m pip install -e .
```

Quick verification:

```bash
bash scripts/predict.sh --help
bash scripts/demo.sh --help
bash -lc 'source scripts/project_env.sh; python run.py --help >/dev/null'
```

## Required external assets

You must provide these yourself:

- FLUX.2-Klein base model directory
- ControlLight LoRA checkpoint
- training dataset, if you want to train

Typical examples:

- `--model-path /path/to/FLUX.2-klein-base-9B`
- `--lora-path /path/to/controllight.safetensors`

## Inference entrypoints

### 1. Predict one image

```bash
bash scripts/predict.sh predict-image \
  --input /path/to/input.jpg \
  --output /path/to/output.png \
  --model-path /path/to/FLUX.2-klein-base-9B \
  --lora-path /path/to/controllight.safetensors \
  --alpha 0.54 \
  --num-inference-steps 20 \
  --guidance-scale 1.0 \
  --seed 42 \
  --device cuda \
  --torch-dtype bfloat16
```

### 2. Fixed four-strength sweep

Runs the public four-strength sweep:

- `0.25`
- `0.50`
- `0.75`
- `1.00`

```bash
bash scripts/predict.sh predict-four \
  --input /path/to/images \
  --output /path/to/out_four \
  --model-path /path/to/FLUX.2-klein-base-9B \
  --lora-path /path/to/controllight.safetensors \
  --num-inference-steps 20 \
  --seed 42 \
  --device cuda \
  --torch-dtype bfloat16
```

### 3. Custom strength sweep

```bash
bash scripts/predict.sh predict-strengths \
  --input /path/to/images \
  --output /path/to/out_strengths \
  --model-path /path/to/FLUX.2-klein-base-9B \
  --lora-path /path/to/controllight.safetensors \
  --alphas 0.20,0.40,0.60,0.80 \
  --num-inference-steps 20 \
  --seed 42 \
  --device cuda \
  --torch-dtype bfloat16
```

`predict-four` and `predict-strengths` accept either:

- a plain image directory
- a `design_materials` root containing `sources/`

Outputs include:

- `manifest.json`
- `sources/<slug>.png`
- `results/<slug>/input.jpg`
- `results/<slug>/grid.jpg`
- `results/<slug>/comparison.gif`
- `results/<slug>/outputs/*.png`
- `results/<slug>/metadata.json`

## Demo

```bash
bash scripts/demo.sh \
  --host 0.0.0.0 \
  --port 7860 \
  --model-path /path/to/FLUX.2-klein-base-9B \
  --lora-path /path/to/controllight.safetensors \
  --device cuda \
  --torch-dtype bfloat16
```

## Training

Main config:

- `config/train_flux2klein_lora.yaml`

Public training scripts:

- `scripts/train/prepare_dataset_links.sh`
- `scripts/train/train_local.sh`
- `scripts/train/train_multigpu.sh`
- `scripts/train/train_watchdog.sh`

Typical usage:

```bash
bash scripts/train/prepare_dataset_links.sh
bash scripts/train/train_local.sh
```

Distributed launch:

```bash
CONFIG=./config/train_flux2klein_lora.yaml \
GPUS=0,1,2,3 \
NUM_PROCESSES=4 \
bash scripts/train/train_multigpu.sh
```

More detail:

- [`scripts/train/README.md`](scripts/train/README.md)

## Repository layout

```text
ControlLight/           Python package and public CLIs
config/                 Public training config
diffusers/              Vendored local diffusers checkout
docs/                   Installation notes
extensions/             Training extensions used by the stack
extensions_built_in/    Built-in model and trainer integrations
jobs/                   Training job launcher code
scripts/                Public shell entrypoints
toolkit/                Core training and inference helpers
predict.py              Public prediction entrypoint
serve_demo.py           Public local demo entrypoint
run.py                  Training launcher
```
