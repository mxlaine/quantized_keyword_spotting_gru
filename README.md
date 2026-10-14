# Quantized Keyword Spotting GRU

An IC design project: a small GRU that recognizes 12 spoken commands (ten
keywords, "unknown" and silence), and how much accuracy it loses when its
weights and activations are compressed to 8-bit integers.

**Short answer: almost none.** The largest model goes from 97.24% to 97.17%
accuracy while its weights shrink 4× (183 KiB → 46 KiB).

## Results

Mean ± standard deviation over three seeds.

<!-- results:start -->
| Hidden size | Parameters | FP32 test accuracy | INT8 simulation test accuracy | Weight storage FP32 → INT8 |
|---:|---:|---:|---:|---:|
| 16 | 5,004 | 94.85 ± 0.11% | 94.59 ± 0.10% | 18.75 → 4.69 KiB |
| 32 | 14,604 | 96.57 ± 0.23% | 96.43 ± 0.29% | 55.50 → 13.88 KiB |
| 48 | 28,812 | 97.11 ± 0.08% | 97.04 ± 0.14% | 110.25 → 27.56 KiB |
| 64 | 47,628 | 97.24 ± 0.16% | 97.17 ± 0.09% | 183.00 → 45.75 KiB |
<!-- results:end -->

![Test accuracy versus hidden size](docs/images/accuracy.png)
![Estimated weight storage](docs/images/weight-storage.png)

Storage counts weights only. INT8 accuracy is simulated in PyTorch, not
measured on hardware.

## How it works

- **Model:** log-mel audio features → two GRU layers → linear classifier,
  trained on Speech Commands v2.
- **Custom GRU:** [`NewGRU`](src/new_gru.py) exposes the internal gate values
  so they can be quantized. Tests check that it matches `torch.nn.GRU`.
- **Quantization:** [post-training quantization](src/ptq/quant_new_gru.py)
  converts weights and activations to INT8 using a small calibration set.
  Biases and arithmetic stay in floating point.

## Quick start

```bash
git clone https://github.com/mxlaine/quantized_keyword_spotting_gru.git
cd quantized_keyword_spotting_gru
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/
```

All 12 trained checkpoints are included in
[`checkpoints/published-sweep/`](checkpoints/published-sweep/). To rerun one
result (downloads the dataset on first use):

```bash
python scripts/reproduce_results.py --hidden-size 64 --seed 0
```

Use `--all` for every configuration, or `--verify-only` to just check the
checkpoint files.

To train and calibrate your own model, see `src/main.py` and
`src/calibrate_ptq.py` (both support `--help`). The exact training settings are
in the [training logs](checkpoints/published-sweep/training-logs/), and SLURM
scripts for cluster sweeps are in [`slurm/`](slurm/).

Trained on Aalto University's Triton cluster (V100 GPUs) with PyTorch 2.8.
