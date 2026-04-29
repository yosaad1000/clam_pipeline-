---
inclusion: always
---

# MXA Pipeline — Known Issues & Findings

## Critical Bug: ONNX / DFP weights do not match the PyTorch checkpoint

### What we found (Apr 29 2026)

We built `compare_extractors.py` to run both `PyTorchResNet50` and `MXAResNet50`
on the same slide and compare the feature vectors they produce before Step 3
(CLAM inference). Results on `6_40x_Raw.ome.tif` (680 patches):

| Metric | Value | Expected |
|--------|-------|----------|
| Cosine similarity (mean) | 0.019 | ~1.0 |
| L2 distance (mean) | 1.40 | ~0.0 |
| Correlation of spatial maps | 0.38 | ~1.0 |

Verdict: **POOR** — the two extractors produce completely different feature vectors.

### Root cause

The ONNX model (`resnet50_trunc.onnx`) and the PyTorch weights
(`checkpoints/resnet50_timm.pth`) are **different checkpoints**. They were never
the same model:

- `resnet50_trunc.onnx` — created Apr 23, exported from a different ResNet50
  (likely the default ImageNet pretrained weights from timm)
- `checkpoints/resnet50_timm.pth` — saved Apr 24, the actual weights used by
  the PyTorch path

Confirmed by directly comparing conv layer weights between the ONNX initializers
and the `.pth` state dict — max absolute difference of **2.79** on the very first
conv layer (should be 0.0 if they were the same model).

The DFP (`resnet50_trunc.dfp`) was compiled correctly from the ONNX — the
compiler is not at fault. The ONNX itself is the wrong model.

### What is NOT broken

- ONNX runtime output matches PyTorch output exactly (cosine sim = 1.0, max diff
  = 0.00005) — the ONNX export process is correct
- The DFP compiles and runs without errors — the MXA hardware and SDK work fine
- Step 1 (patch extraction) is unaffected
- Step 3 (CLAM inference) is unaffected once it receives correct features

### Fix required

Re-export the ONNX from `checkpoints/resnet50_timm.pth` and recompile the DFP:

```python
# export_onnx.py — run on any machine with torch + timm
import torch, timm

model = timm.create_model(
    "resnet50", features_only=True, out_indices=(3,),
    pretrained=False, num_classes=0
)
state = torch.load("checkpoints/resnet50_timm.pth", map_location="cpu")
model.load_state_dict(state, strict=False)
model.eval()

pool = torch.nn.AdaptiveAvgPool2d(1)

class Wrapped(torch.nn.Module):
    def forward(self, x):
        return pool(model(x)[0]).squeeze(-1).squeeze(-1)  # [N, 1024]

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    Wrapped(), dummy, "resnet50_trunc.onnx",
    input_names=["input"], output_names=["features"],
    dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
    opset_version=13,
)
```

Then on the Pi:
```bash
bash compile_resnet.sh   # recompiles resnet50_trunc.onnx -> resnet50_trunc.dfp
```

After recompiling, re-run `compare_extractors.py` to verify cosine similarity
is > 0.99 before using the MXA path in production.

---

## MXA vs PyTorch Speed (measured Apr 29 2026)

On Raspberry Pi, 680 patches (256x256), batch_size=32:

| Extractor | Time | Throughput |
|-----------|------|------------|
| PyTorchResNet50 (CPU) | 168s | ~4 patches/s |
| MXAResNet50 | 4s | ~170 patches/s |

MXA is **~42x faster** than PyTorch CPU on the Pi. Once the weight mismatch is
fixed, the MXA path should be the default for production use.

---

## Tooling added

- `pi_deploy/compare_extractors.py` — runs both extractors on the same slide,
  computes cosine similarity / L2 / L1 distance per patch, saves
  `diff_report.json` and `diff_histogram.png`. Use this after any DFP recompile
  to validate alignment before running Step 3.

---

## Architecture reference

Both extractors target the same feature space:

- ResNet50 truncated at `layer3` -> spatial map `[N, 1024, 14, 14]`
- Global Average Pooling -> `[N, 1024]`
- L2 normalisation per vector
- Transform: `Resize(224) -> ToTensor() -> Normalize(ImageNet)` — no CenterCrop

The PyTorch path uses `timm.create_model(features_only=True, out_indices=(3,))`
with an explicit `AdaptiveAvgPool2d(1)`. The MXA path applies GAP manually on
the DFP output since the pooling layer is baked into the ONNX export.
