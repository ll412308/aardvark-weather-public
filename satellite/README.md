# 1BAMUA local SetConv autoencoder

This package is deliberately independent of Aardvark's ViT, forecast processor,
U-Net, and multi-instrument paths.

Aardvark's `convDeepSet` constructs separate longitude and latitude kernel
matrices and combines them with dense `einsum` operations. Its `OffToOn` line is
`...cw,...wx,...wy->...cxy`. That representation is useful for the layouts used
by Aardvark, but is not a scalable general operator for hundreds of thousands of
paired `(lon_i, lat_i)` observations.

Here each observation is kept paired. It contributes to a configurable local
neighbourhood (`3 x 3` by default) using a learnable Gaussian, followed by
`scatter_add`. Longitude indices wrap periodically and latitude indices are
clipped at the poles. Both directions therefore use `O(B N K)` work and memory,
where `K` is the small fixed number of neighbours.

Run from the repository root:

```powershell
python -m satellite.test_bamua
python -m satellite.test_bamua --zarr "F:\lyh_data\data_zarr_no_provider_filter\1bamua.zarr" --context 100 --target 20 --target-overlap 0.5
python -m satellite.train_bamua --config satellite/configs/bamua_smoke.yaml
```

`--target-overlap` controls how many target points are also present in the
context set. `0.0` gives fully held-out targets, `1.0` gives ordinary
autoencoder-style targets sampled from context, and `0.5` mixes both signals.

Resume the newest saved checkpoint for the same `run_name`:

```powershell
python -m satellite.train_bamua --config satellite/configs/bamua_smoke.yaml --resume
```

Or resume a specific checkpoint:

```powershell
python -m satellite.train_bamua --config satellite/configs/bamua_smoke.yaml --resume "runs/bamua_smoke/20260825_120000/latest.pth"
```

Each run directory contains `loss_log.jsonl`, `best.pth`, periodic
`epoch_XXXX.pth`/`latest.pth`, and `reconstructions/epoch_XXXX.pth`. A
reconstruction file contains the first validation batch's prediction, target,
validity mask, longitude, latitude, and sample time.

The YAML also controls reproducibility, automatic mixed precision, and the
learning-rate schedule:

```yaml
train:
  seed: 42
  deterministic: false
  mixed_precision: true
  amp_dtype: "float16"

scheduler:
  name: "warmup_cosine"
  warmup_epochs: 2
  min_lr: 1.0e-6
```

`warmup_cosine` is updated after every optimizer step. It linearly raises the
learning rate during warmup and then follows a cosine curve down to `min_lr`.
Mixed precision is enabled only when CUDA is available; CPU training falls back
to float32 automatically. Model, optimizer, scheduler, AMP scaler, loss history,
early-stopping state, and random-number-generator states are all checkpointed.

Complete 6-hour bins are sorted by Zarr `time_series`: early bins are used for
training, the following bins for validation, and the latest bins for testing.
The final test reconstructs every observation in a selected bin three times,
using 100%, 90%, and 80% of its observations as encoder context. Context
encoding and all-point decoding are chunked to control GPU memory. Run only the
full reconstruction test from the newest checkpoint with:

```powershell
python -m satellite.train_bamua --config satellite/configs/bamua_smoke.yaml --test-only
```

Results are written under `test_reconstructions/sample_XXXX/`. `metrics.json`
reports RMSE over every query, context points, held-out points, and each BT
channel. `target_all.pth` is stored once, while each context percentage has its
own prediction file.
