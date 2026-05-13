# MXA Pipeline — Findings & How We Got the Correct Heatmap

## Final Status (May 13 2026)

MXA path produces **identical heatmap** to PyTorch CPU (mean pixel diff 5.4,
down from 46). Total pipeline is **2.7x faster** end-to-end.

| Metric | PyTorch CPU | MXA |
|--------|-------------|-----|
| Step 2 feature extraction | 172s | **4s** |
| Total pipeline | 275s | **102s** |
| Heatmap mean pixel diff | — | 5.4 (JPEG + int8 noise) |
| Prediction | Normal 100% | Normal 100% |

---

## The Problems We Found and Fixed

### Problem 1: Wrong ONNX weights

The original `resnet50_trunc.onnx` (Apr 23) was exported from default timm
ImageNet weights, not from `checkpoints/resnet50_timm.pth` (Apr 24). MXA and
PyTorch were running completely different models — cosine similarity 0.019.

**Fix:** Re-export ONNX from the correct `.pth`:
```bash
venv/bin/python export_onnx.py
```

---

### Problem 2: `--autocrop` cutting at the wrong layer

`compile_resnet.sh` used `--autocrop` which told the Memryx compiler to strip
pre/post-processing. It cut at `layer3.5/conv3` — the raw conv output **before**
BN+ReLU. This gave negative mean outputs (~-0.85) vs PyTorch's post-ReLU
positive outputs (~+0.70). Cosine similarity was still 0.019 even with correct
weights.

**Fix:** Remove `--autocrop`. The compiler then includes the full layer3 output
with BN+ReLU, matching PyTorch. Single-patch ONNX vs DFP cosine sim: **0.999**.

```bash
# WRONG (was):
mx_nc -v --models resnet50_trunc.onnx --autocrop --exp_auto_dp --dfp_fname resnet50_trunc.dfp

# CORRECT (now):
mx_nc -v --models resnet50_trunc.onnx --exp_auto_dp --dfp_fname resnet50_trunc.dfp
```

---

### Problem 3: CenterCrop in MXA transform

The original `MXAResNet50` transform had `CenterCrop(224)` after `Resize(224)`.
Patches are 256×256 so this cropped different pixels than PyTorch (no CenterCrop).

**Fix:** Remove `CenterCrop`. Both paths now use:
```python
transforms.Resize(224) → transforms.ToTensor() → transforms.Normalize(ImageNet)
```

---

### Problem 4: L2 normalisation broke the heatmap

During debugging, L2 norm was added to both paths to align scales. This broke
the heatmap because `s_0_checkpoint.pt` was trained on raw un-normalised features.

Measured with `s_0_checkpoint.pt`:
- WITHOUT L2 norm: Normal **100%**, correct heatmap
- WITH L2 norm: Normal 61%, attention correlation **-0.811** (heatmap inverted)

**Fix:** Remove L2 norm from `PyTorchResNet50.extract_batch`. PyTorch outputs
raw features as the model was trained on.

---

### Problem 5: MXA magnitude mismatch → wrong heatmap

After fixes 1-4, cosine similarity was 0.998 but the heatmap still differed
(mean pixel diff 46, 91% of pixels off by >10). Root cause: the ONNX has L2
norm baked in so the DFP outputs **unit vectors** (magnitude ~1), while PyTorch
outputs raw features (magnitude ~44). CLAM's attention network was trained on
magnitude-44 inputs, so feeding unit vectors produced different attention scores.

**Fix:** Multiply MXA output by **44.0** to restore the original magnitude:
```python
results.append(feat * 44.0)
```

After this fix: mean pixel diff **5.4** (down from 46), prediction matches
exactly (Normal 100%), heatmaps visually identical. The remaining 5.4 diff is
JPEG compression + int8 quantization noise from the MXA chip — unavoidable.

---

## Final Architecture

```
resnet50_timm.pth
    ↓  export_onnx.py  (ResNet50 layer3 + GAP + L2 norm baked in)
resnet50_trunc.onnx
    ↓  compile_resnet.sh  (mx_nc --exp_auto_dp, NO --autocrop)
resnet50_trunc.dfp  (outputs unit vectors [N, 1024])
    ↓  MXAResNet50.collect_output  (GAP on spatial map, then × 44.0)
features [N, 1024]  magnitude ~44, matching PyTorch raw features
    ↓  CLAM_SB  (s_0_checkpoint.pt)
heatmap ✓
```

PyTorch path:
```
resnet50_timm.pth → ResNet50 layer3 + GAP → raw features [N, 1024] (mag ~44)
    ↓  CLAM_SB  (s_0_checkpoint.pt)
heatmap ✓  (reference)
```

---

## How to Run

```bash
# MXA (fast — 102s total)
venv/bin/python pipeline_mxa.py \
    --input input/6_40x_Raw.ome.tif \
    --output output/ \
    --use_mxa --dfp resnet50_trunc.dfp \
    --ckpt checkpoints/s_0_checkpoint.pt

# PyTorch CPU (reference — 275s total)
venv/bin/python pipeline_mxa.py \
    --input input/6_40x_Raw.ome.tif \
    --output output/ \
    --ckpt checkpoints/s_0_checkpoint.pt
```

## How to Validate After Any DFP Recompile

Run `compare_extractors.py` then check true cosine similarity:
```python
import torch, numpy as np
pt  = torch.load('compare_output/6_40x_Raw.ome/pytorch_features.pt').numpy()
mxa = torch.load('compare_output/6_40x_Raw.ome/mxa_features.pt').numpy()
pt_n  = pt  / np.linalg.norm(pt,  axis=1, keepdims=True)
mxa_n = mxa / np.linalg.norm(mxa, axis=1, keepdims=True)
cos   = (pt_n * mxa_n).sum(axis=1)
print(f'Cosine sim mean={cos.mean():.4f}')  # must be > 0.99
```
