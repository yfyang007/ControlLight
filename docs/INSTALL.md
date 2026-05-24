# Installation Guide

This project contains two local components:

- a vendored `diffusers/` checkout for prediction
- vendored training code (`run.py`, `jobs/`, `toolkit/`, `extensions/`)

Do not rely on upstream PyPI `diffusers` alone.

## Install

```bash
python -m pip install --upgrade pip
python -m pip install -e diffusers
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Verify

```bash
bash scripts/predict.sh --help
bash scripts/demo.sh --help
bash -lc 'source scripts/project_env.sh; python run.py --help >/dev/null'
```

## Recommended public entrypoints

- prediction CLI: `bash scripts/predict.sh ...`
- demo server: `bash scripts/demo.sh ...`
- training scripts: `bash scripts/train/*.sh`

## Recommended environment

Use the same environment for both prediction and training:

- `aitoolkit`

The shared bootstrap script is:

- `scripts/project_env.sh`

## Common failure modes

### Wrong `diffusers`

Symptom:

```text
ImportError: cannot import name 'ControlLightPipeline' from 'diffusers'
```

Fix:

```bash
python -m pip install -e /path/to/controllight/diffusers
```

### Mixed `python` and `pip`

Symptom:

- install succeeds, import still fails

Fix:

- use `python -m pip ...` consistently in the same environment

### Missing model assets

You still need access to:

- the base model path passed to `--model-path`
- the LoRA checkpoint passed to `--lora-path`

### `torchao` / `peft` mismatch

Symptom:

```text
ImportError: Found an incompatible version of torchao
```

Fix:

- use a known-good environment for prediction and training
- this repository keeps local vendored integrations specifically to reduce upstream drift
