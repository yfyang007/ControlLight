# ControlLight Landing Page

Static homepage prototype for ControlLight.

The current interaction focuses on one sample's inference-strength curve:

```text
input -> alpha 0.25 -> alpha 0.50 -> alpha 0.75 -> alpha 1.00
```

## Run locally

From the repo root:

```bash
cd /data/yfyang/controllight/landing
python3 -m http.server 8015
```

Then open:

```text
http://127.0.0.1:8015
```

The page reads showcase assets through:

```text
./assets -> ../outputs/realir_bench_lowlight_prompt_alpha_gif
```
