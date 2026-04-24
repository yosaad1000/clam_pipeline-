"""
pipeline_single.py — CLAM pipeline for a single WSI image.

Usage:
    conda run -n clam python pipeline_single.py \
        --input  path/to/slide.tif \
        --output path/to/output_dir \
        [--patch_size 256] \
        [--batch_size 256] \
        [--alpha 0.4] \
        [--ckpt CLAM/heatmaps/demo/ckpts/s_0_checkpoint.pt]
"""
import os, sys, time, json, logging, traceback, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import h5py
import openslide
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, "CLAM")
from models.model_clam import CLAM_SB, CLAM_MB
from wsi_core.WholeSlideImage import WholeSlideImage

# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="CLAM single-slide pipeline")
    p.add_argument("--input",      required=True,  help="Path to input WSI (.tif / .svs)")
    p.add_argument("--output",     required=True,  help="Output directory")
    p.add_argument("--ckpt",       default="CLAM/heatmaps/demo/ckpts/s_0_checkpoint.pt")
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--alpha",      type=float, default=0.4, help="Heatmap blend alpha")
    p.add_argument("--model_name", type=str, default="resnet50_trunc",
                   choices=["resnet50_trunc", "uni_v1", "conch_v1"],
                   help="Feature extractor model")
    p.add_argument("--uni_ckpt",   type=str, default=None,
                   help="Path to UNI weights (required when --model_name uni_v1)")
    p.add_argument("--conch_ckpt", type=str, default=None,
                   help="Path to CONCH weights (required when --model_name conch_v1)")
    return p.parse_args()

# ── Logger ────────────────────────────────────────────────────────────────────
def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(log_path.stem)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def save_meta(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

# ── Step 1 ────────────────────────────────────────────────────────────────────
def step1(slide_path: Path, out: Path, patch_size: int, logger) -> dict:
    import subprocess
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("STEP 1 — Tissue Segmentation & Patch Extraction")
    logger.info("=" * 60)

    slide_name = slide_path.stem
    slide_ext  = slide_path.suffix
    step1_dir  = out / "step1_patches"
    for d in ["patches", "masks", "stitches"]:
        (step1_dir / d).mkdir(parents=True, exist_ok=True)

    process_csv = step1_dir / "process_list.csv"
    with open(process_csv, "w") as f:
        f.write("slide_id,process,status,seg_level,sthresh,mthresh,close,use_otsu,"
                "keep_ids,exclude_ids,a_t,a_h,max_n_holes,vis_level,line_thickness,"
                "use_padding,contour_fn\n")
        f.write(f"{slide_name}{slide_ext},1,tbp,-1,8,7,4,False,none,none,"
                f"100.0,16.0,8,-1,250,True,four_pt\n")

    cmd = (f'conda run -n clam python CLAM/create_patches_fp.py '
           f'--source "{slide_path.parent.resolve()}" '
           f'--save_dir "{step1_dir.resolve()}" '
           f'--patch_size {patch_size} --step_size {patch_size} '
           f'--seg --patch --stitch '
           f'--process_list "{process_csv.resolve()}"')
    logger.info(f"CMD: {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    logger.debug("STDOUT:\n" + r.stdout)
    if r.returncode != 0:
        logger.error("STDERR:\n" + r.stderr)
        raise RuntimeError("Step 1 failed")

    patch_h5 = step1_dir / "patches" / f"{slide_name}.h5"
    patch_count = 0
    if patch_h5.exists():
        with h5py.File(patch_h5, "r") as f:
            patch_count = len(f["coords"][:])
    logger.info(f"Patches extracted: {patch_count}  ({time.time()-t0:.2f}s)")

    meta = {"step": 1, "name": "Patch Extraction",
            "output": {"patch_h5": str(patch_h5), "patch_count": patch_count},
            "timing_seconds": round(time.time()-t0, 3)}
    save_meta(out / "step1_metadata.json", meta)
    return meta

# ── Step 2 ────────────────────────────────────────────────────────────────────
def step2(slide_path: Path, out: Path, batch_size: int, logger,
          model_name: str = "resnet50_trunc", uni_ckpt: str = None,
          conch_ckpt: str = None) -> dict:
    import subprocess
    t0 = time.time()
    logger.info("=" * 60)
    logger.info(f"STEP 2 — Feature Extraction ({model_name})")
    logger.info("=" * 60)

    slide_name = slide_path.stem
    slide_ext  = slide_path.suffix
    step1_dir  = out / "step1_patches"
    step2_dir  = out / "step2_features"
    for d in ["pt_files", "h5_files"]:
        (step2_dir / d).mkdir(parents=True, exist_ok=True)

    process_csv = step1_dir / "process_list.csv"

    env_prefix = ""
    if model_name == "uni_v1":
        if not uni_ckpt:
            raise ValueError("--uni_ckpt must be set when using uni_v1")
        env_prefix = f'UNI_CKPT_PATH="{Path(uni_ckpt).resolve()}" '
    elif model_name == "conch_v1":
        if not conch_ckpt:
            conch_ckpt = os.environ.get("CONCH_CKPT_PATH", "")
        if not conch_ckpt:
            raise ValueError("--conch_ckpt must be set when using conch_v1")
        env_prefix = f'CONCH_CKPT_PATH="{Path(conch_ckpt).resolve()}" '

    cmd = (f'{env_prefix}conda run -n clam python CLAM/extract_features_fp.py '
           f'--data_h5_dir "{step1_dir.resolve()}" '
           f'--data_slide_dir "{slide_path.parent.resolve()}" '
           f'--slide_ext {slide_ext} '
           f'--csv_path "{process_csv.resolve()}" '
           f'--feat_dir "{step2_dir.resolve()}" '
           f'--model_name {model_name} '
           f'--batch_size {batch_size} '
           f'--target_patch_size 224')
    logger.info(f"CMD: {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    logger.debug("STDOUT:\n" + r.stdout)
    if r.returncode != 0:
        logger.error("STDERR:\n" + r.stderr[-3000:])
        raise RuntimeError("Step 2 failed")

    pt_path = step2_dir / "pt_files" / f"{slide_name}.pt"
    feat_shape = None
    if pt_path.exists():
        feats = torch.load(pt_path, map_location="cpu")
        feat_shape = list(feats.shape)
    logger.info(f"Features: {feat_shape}  ({time.time()-t0:.2f}s)")

    meta = {"step": 2, "name": "Feature Extraction",
            "output": {"pt_file": str(pt_path), "feature_shape": feat_shape},
            "timing_seconds": round(time.time()-t0, 3)}
    save_meta(out / "step2_metadata.json", meta)
    return meta

# ── Step 3 ────────────────────────────────────────────────────────────────────
def step3(slide_path: Path, out: Path, ckpt_path: Path,
          patch_size: int, alpha: float, logger, embed_dim: int = 1024) -> dict:
    from scipy.stats import percentileofscore
    import tifffile as _tifffile
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("STEP 3 — CLAM Inference + Heatmap Generation")
    logger.info("=" * 60)

    slide_name = slide_path.stem
    step2_dir  = out / "step2_features"
    step3_dir  = out / "step3_heatmaps"
    step3_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # load features
    pt_path = step2_dir / "pt_files" / f"{slide_name}.pt"
    features = torch.load(pt_path, map_location=device)
    logger.info(f"Features: {list(features.shape)}")

    # load model — auto-detect CLAM_SB vs CLAM_MB from checkpoint keys
    state = torch.load(ckpt_path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in state.items()
             if not k.startswith("instance_loss_fn")}
    is_mb = isinstance(state.get("classifiers.0.weight"), torch.Tensor) and \
            isinstance(state.get("classifiers.1.weight"), torch.Tensor)
    ModelClass = CLAM_MB if is_mb else CLAM_SB
    logger.info(f"Model type: {'CLAM_MB' if is_mb else 'CLAM_SB'}")
    model = ModelClass(gate=True, size_arg="small", dropout=0.25,
                       n_classes=2, embed_dim=embed_dim).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # inference
    with torch.no_grad():
        logits, Y_prob, Y_hat, A_raw, _ = model(features)
    Y_hat_val = Y_hat.item()
    probs = Y_prob.squeeze().cpu().numpy()
    attn_raw = A_raw.squeeze().cpu().numpy()
    # CLAM_MB returns [n_classes, N]; take the predicted class row
    attn = attn_raw[Y_hat_val] if attn_raw.ndim == 2 else attn_raw
    label_map = {0: "Normal", 1: "Tumor"}
    logger.info(f"Prediction: {label_map[Y_hat_val]}  Normal={probs[0]:.4f}  Tumor={probs[1]:.4f}")

    # save prediction
    with open(step3_dir / "prediction.txt", "w") as f:
        f.write(f"Slide     : {slide_name}\n")
        f.write(f"Prediction: {label_map[Y_hat_val]} (class {Y_hat_val})\n")
        f.write(f"Normal    : {probs[0]:.6f}\n")
        f.write(f"Tumor     : {probs[1]:.6f}\n")

    # load coords
    h5_path = step2_dir / "h5_files" / f"{slide_name}.h5"
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]

    # normalize attention to percentiles
    attn_pct = np.array([percentileofscore(attn, s) for s in attn])

    # determine vis_level — force 0 for TiffSlide fallback (no real pyramid)
    wsi_obj = WholeSlideImage(str(slide_path))
    try:
        _wsi = openslide.open_slide(str(slide_path))
        _seg_level = _wsi.get_best_level_for_downsample(64)
        _vis_level = -1
        _wsi.close()
        logger.info(f"Backend: openslide  seg_level={_seg_level}")
    except Exception:
        _seg_level = 0
        _vis_level = 0
        logger.info("Backend: TiffSlide  vis_level=0 (full resolution forced)")

    wsi_obj.segmentTissue(
        seg_level=_seg_level, sthresh=8, mthresh=7, close=4, use_otsu=False,
        filter_params={"a_t": 100, "a_h": 16, "max_n_holes": 8}
    )

    # generate heatmap
    heatmap_pil = wsi_obj.visHeatmap(
        scores=attn_pct, coords=coords, vis_level=_vis_level,
        patch_size=(patch_size, patch_size), alpha=alpha,
        blur=True, convert_to_percentiles=False,
        cmap="jet", blank_canvas=False, segment=True,
    )
    heatmap_size = list(heatmap_pil.size)
    logger.info(f"Heatmap size: {heatmap_size}")

    # save JPG preview
    jpg_path = step3_dir / f"{slide_name}_heatmap.jpg"
    heatmap_pil.save(str(jpg_path))

    # save full-res TIFF
    tiff_path = step3_dir / f"{slide_name}_heatmap.tiff"
    _tifffile.imwrite(str(tiff_path), np.array(heatmap_pil),
                      photometric="rgb", compression="deflate",
                      metadata={"axes": "YXS"})
    logger.info(f"Saved TIFF: {tiff_path}  size={heatmap_size}")

    # save original slide thumbnail
    wsi_raw = wsi_obj.getOpenSlide()
    best_level = wsi_raw.get_best_level_for_downsample(32) if _vis_level != 0 else 0
    thumb_size = wsi_raw.level_dimensions[best_level]
    orig = wsi_raw.read_region((0, 0), best_level, thumb_size).convert("RGB")
    orig.save(str(step3_dir / f"{slide_name}_original.jpg"))

    # save top-15 patches
    top_patches_dir = step3_dir / "top_patches"
    top_patches_dir.mkdir(exist_ok=True)
    top_ids = np.argsort(attn)[-15:][::-1]
    for rank, idx in enumerate(top_ids):
        x, y = int(coords[idx][0]), int(coords[idx][1])
        score = float(attn[idx])
        region = wsi_raw.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
        region.save(str(top_patches_dir / f"{rank:02d}_x{x}_y{y}_s{score:.4f}.png"))
    logger.info(f"Saved top-15 patches → {top_patches_dir}/")

    elapsed = time.time() - t0
    logger.info(f"Step 3 completed in {elapsed:.2f}s")

    meta = {
        "step": 3, "name": "Inference + Heatmap",
        "inference": {
            "prediction": label_map[Y_hat_val],
            "prob_normal": float(probs[0]), "prob_tumor": float(probs[1]),
        },
        "output": {
            "heatmap_jpg": str(jpg_path),
            "heatmap_tiff": str(tiff_path),
            "heatmap_size_wh": heatmap_size,
            "original_jpg": str(step3_dir / f"{slide_name}_original.jpg"),
            "top_patches_dir": str(top_patches_dir),
        },
        "timing_seconds": round(elapsed, 3),
    }
    save_meta(out / "step3_metadata.json", meta)
    return meta

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    slide_path = Path(args.input)
    out        = Path(args.output) / slide_path.stem
    ckpt_path  = Path(args.ckpt)

    if not slide_path.exists():
        print(f"[ERROR] Input not found: {slide_path}")
        sys.exit(1)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out / f"{slide_path.stem}.log")

    t_start = time.time()
    logger.info(f"Input  : {slide_path}")
    logger.info(f"Output : {out}")

    try:
        m1 = step1(slide_path, out, args.patch_size, logger)
        m2 = step2(slide_path, out, args.batch_size, logger,
                   model_name=args.model_name, uni_ckpt=args.uni_ckpt,
                   conch_ckpt=args.conch_ckpt)
        m3 = step3(slide_path, out, ckpt_path, args.patch_size, args.alpha, logger,
                   embed_dim=512 if args.model_name == "conch_v1" else 1024)
        status = "SUCCESS"
    except Exception as e:
        logger.error(f"Pipeline FAILED: {e}\n{traceback.format_exc()}")
        status = f"FAILED: {e}"

    total = time.time() - t_start
    summary = {
        "slide": str(slide_path), "status": status,
        "total_time_seconds": round(total, 3),
        "output_dir": str(out),
    }
    save_meta(out / "summary.json", summary)
    logger.info(f"Status: {status}  Total: {total:.2f}s")

if __name__ == "__main__":
    main()
