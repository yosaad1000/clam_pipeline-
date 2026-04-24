# Camelyon16 WSI Tumor Detection Pipeline

End-to-end pipeline for whole slide image (WSI) tumor detection using [CLAM](https://github.com/mahmoodlab/CLAM) on the Camelyon16 dataset.

## Overview

Given a WSI (`.tif`), the pipeline:
1. Segments tissue and extracts patches
2. Encodes each patch into feature vectors using a pretrained ResNet50
3. Runs a MIL (Multiple Instance Learning) classifier to predict tumor vs normal
4. Generates an attention heatmap overlaid on the slide showing which regions drove the prediction

## Workspace Structure

```
HIDA/
├── archive (1)/              # Raw WSI data (Camelyon16)
│   ├── tumor_001.tif         # Input slide
│   └── tumor_001_mask.tif    # Ground truth mask
│
├── patches/                  # Step 1 outputs
│   ├── masks/                # Tissue segmentation visualizations
│   ├── patches/              # Patch coordinates (.h5)
│   ├── stitches/             # Patch location thumbnails
│   └── process_list_autogen.csv
│
├── CLAM/                     # CLAM repo + our scripts
│   ├── create_patches_fp.py  # Step 1: patching
│   ├── extract_features_fp.py # Step 2: feature extraction
│   ├── create_heatmaps.py    # Step 3: heatmap generation
│   ├── infer_clam.py         # Step 3: single-slide inference (our script)
│   ├── features/             # Step 2 outputs
│   │   ├── pt_files/         # Feature tensors [N x 1024] (.pt)
│   │   └── h5_files/         # Features + coords (.h5)
│   └── heatmaps/
│       ├── configs/tumor_001.yaml          # Our heatmap config
│       ├── demo/ckpts/s_0_checkpoint.pt    # Pretrained CLAM checkpoint
│       ├── process_lists/tumor_001_list.csv
│       ├── heatmap_production_results/     # Final heatmap JPGs
│       └── heatmap_raw_results/            # Raw attention blockmaps
│
└── README.md
```

## Environment

```bash
conda activate transmil
```

Key packages: `torch 1.10.1+cu111`, `openslide-python`, `h5py`, `timm 0.6.13`, `pytorch-lightning`

## Pipeline Steps

### Step 1 — Patch Extraction

Segments tissue from background and extracts 256×256 patches.

```bash
python CLAM/create_patches_fp.py \
  --source "archive (1)" \
  --save_dir patches \
  --patch_size 256 \
  --step_size 256 \
  --seg --patch --stitch \
  --process_list process_list_autogen.csv
```

Output: `patches/patches/tumor_001.h5` — 36,710 patch coordinates  
Visualization: `patches/masks/tumor_001.jpg` (tissue contours), `patches/stitches/tumor_001.jpg` (patch locations)

---

### Step 2 — Feature Extraction

Runs each patch through a pretrained ResNet50 to produce 1024-dim feature vectors.

```bash
python CLAM/extract_features_fp.py \
  --data_h5_dir patches \
  --data_slide_dir "archive (1)" \
  --slide_ext .tif \
  --csv_path patches/process_list_autogen.csv \
  --feat_dir CLAM/features \
  --model_name resnet50_trunc \
  --batch_size 256 \
  --target_patch_size 224
```

Output: `CLAM/features/pt_files/tumor_001.pt` — tensor of shape `[36710, 1024]`

---

### Step 3 — Inference + Heatmap

#### Quick inference (single slide)

```bash
cd CLAM
python infer_clam.py \
  --pt_file features/pt_files/tumor_001.pt \
  --ckpt heatmaps/demo/ckpts/s_0_checkpoint.pt
```

#### Full heatmap generation

```bash
cd CLAM
python create_heatmaps.py --config_file tumor_001.yaml
```

Output:
- `heatmaps/heatmap_production_results/tumor_001_heatmap/tumor_tissue/` — blended heatmap on slide
- `heatmaps/heatmap_raw_results/tumor_001_heatmap/` — raw attention blockmap
- `heatmaps/heatmap_production_results/tumor_001_heatmap/sampled_patches/` — top-15 highest attention patches

---

## Heatmap Color Guide

| Color | Meaning |
|-------|---------|
| Red / Yellow | High attention — model focused here (tumor region with correct weights) |
| Green | Medium attention |
| Blue | Low attention — model ignored these areas |
| Black | Background (no tissue) |

> **Note:** The bundled checkpoint (`s_0_checkpoint.pt`) was trained on TCGA lung cancer data, not Camelyon16. For meaningful tumor localization on lymph node slides, retrain CLAM on the full Camelyon16 dataset.

## Training from Scratch

To train on the full Camelyon16 dataset:
1. Download all slides from [camelyon16.grand-challenge.org](https://camelyon16.grand-challenge.org/Data/)
2. Run Steps 1 and 2 on all slides
3. Create splits: `python CLAM/create_splits_seq.py`
4. Train: `python CLAM/main.py --config your_config.yaml`

## Classes

| Label | Class |
|-------|-------|
| 0 | Normal tissue |
| 1 | Tumor tissue |
