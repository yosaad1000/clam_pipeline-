# CLAM Pipeline — Raspberry Pi + Memryx MXA Deployment

## What's in this package

```
pi_deploy/
├── pipeline_mxa.py          # Main pipeline (single or batch)
├── resnet50_trunc.onnx      # ResNet50 encoder (compile this to .dfp)
├── compile_resnet.sh        # Compiles ONNX → DFP for MXA
├── setup_pi.sh              # Installs all dependencies
├── checkpoints/
│   └── camelyon_s0.pt       # CLAM classifier (Camelyon16+17, lymph node)
├── CLAM/                    # Core CLAM modules
└── input/                   # Drop your WSI slides here
```

## Setup (run once on the Pi)

```bash
bash setup_pi.sh
```

Then install the Memryx SDK `.whl` from https://developer.memryx.com

## Compile ResNet50 for MXA (run once)

```bash
bash compile_resnet.sh
```

This generates `resnet50_trunc.dfp` — the compiled model for the accelerator.

## Run the pipeline

Single slide:
```bash
python pipeline_mxa.py \
    --input input/slide.tif \
    --output output/
```

Batch (whole folder):
```bash
python pipeline_mxa.py \
    --input input/ \
    --output output/ \
    --batch
```

With custom checkpoint:
```bash
python pipeline_mxa.py \
    --input input/ \
    --output output/ \
    --batch \
    --ckpt checkpoints/camelyon_s0.pt \
    --dfp resnet50_trunc.dfp
```

## All arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--input` | required | Slide file or directory |
| `--output` | required | Output directory |
| `--batch` | False | Treat input as directory |
| `--ext` | `.tif` | Slide extension for batch mode |
| `--ckpt` | `checkpoints/camelyon_s0.pt` | CLAM classifier checkpoint |
| `--dfp` | `resnet50_trunc.dfp` | Memryx DFP file |
| `--patch_size` | 256 | Patch size in pixels |
| `--batch_size` | 64 | Patches per MXA batch |
| `--alpha` | 0.4 | Heatmap blend transparency |

## How it works

```
WSI slide
   ↓
Step 1: Tissue segmentation + patch extraction (CPU)
   ↓
Step 2: ResNet50 feature extraction (Memryx MXA ← accelerated)
   ↓
Step 3: CLAM attention inference + heatmap generation (CPU)
   ↓
output/<slide_name>/
   ├── step3_heatmaps/<slide>_heatmap.jpg   ← attention heatmap
   ├── step3_heatmaps/<slide>_original.jpg  ← slide thumbnail
   ├── step3_heatmaps/top_patches/          ← top 15 attended patches
   ├── step3_heatmaps/prediction.txt        ← Tumor/Normal + probabilities
   └── summary.json
```

## Checkpoints available

| File | Tissue | Task |
|------|--------|------|
| `camelyon_s0.pt` | Lymph node | Tumor vs Normal (Camelyon16+17) |

Add more checkpoints to `checkpoints/` and pass with `--ckpt`.
