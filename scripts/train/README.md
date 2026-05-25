# Training Scripts

This folder now exposes a small set of public, easier-to-read entrypoints.

## Recommended entrypoints

| Purpose | Script |
|---|---|
| Activate the shared project environment | `scripts/project_env.sh` |
| Run prediction in the shared environment | `scripts/predict.sh` |
| Run the local demo in the shared environment | `scripts/demo.sh` |
| Launch distributed training | `scripts/train/train_multigpu.sh` |
| Start the standard local training run | `scripts/train/train_local.sh` |
| Keep training alive with automatic resume | `scripts/train/train_watchdog.sh` |
| Python watchdog implementation | `scripts/train/watchdog_resume.py` |

## Unified environment

Prediction and training now default to the same conda environment:

- `controlight`

The shared bootstrap lives at:

- `scripts/project_env.sh`

## Dataset layout

Prepare your dataset yourself under the paths referenced by the training config.
The default public config expects a structure like:

```text
datasets/flux2klein_alpha_interp5_20260501_unified_edgeexp/
  control/
  mask_normrgb_l01/
  mask_normrgb_l02/
  mask_normrgb_l03/
  mask_normrgb_l04/
  mask_normrgb_l05/
  target_l01/
  target_l02/
  target_l03/
  target_l04/
  target_l05/
```

If you use a different dataset root, duplicate `config/train_flux2klein_lora.yaml` and update the dataset paths there.

## Typical workflow

### 1) Start the main training job

```bash
bash scripts/train/train_local.sh
```

### 2) Or run the watchdog instead

```bash
bash scripts/train/train_watchdog.sh
```

## Common overrides

All scripts accept configuration through environment variables. The most useful ones are:

- `REPO_ROOT`
- `CONFIG`
- `RUN_NAME`
- `CUDA_VISIBLE_DEVICES` or `GPUS`
- `NUM_PROCESSES`
- `MAIN_PROCESS_PORT`
- `CONDA_ENV` and `CONDA_SH` if you need to override the default `controlight` environment

Example:

```bash
RUN_NAME=my_experiment \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_PROCESSES=4 \
bash scripts/train/train_local.sh
```
