"""
pipeline_mxa.py — CLAM pipeline for Raspberry Pi (with optional MXA acceleration).

Usage (single slide):
    python pipeline_mxa.py --input slide.tif --output output_dir

Usage (batch):
    python pipeline_mxa.py --input slides_dir --output output_dir --batch

Args:
    --input       Path to a single WSI file OR a directory of WSIs
    --output      Output directory
    --batch       Flag: treat --input as a directory (batch mode)
    --ext         Slide extension when in batch mode (default: .tif)
    --ckpt        CLAM classifier checkpoint (default: checkpoints/s_0_checkpoint.pt)
    --patch_size  Patch size in pixels (default: 256)
    --batch_size  Feature extraction batch size (default: 256)
    --alpha       Heatmap blend alpha (default: 0.4)
    --use_mxa     Use Memryx MXA for feature extraction (default: False, uses PyTorch CPU)
    --dfp         Memryx DFP file for ResNet50 (default: resnet50_trunc.dfp, only used with --use_mxa)
    --tiff        Also save full-res TIFF heatmap
"""
import os, sys, time, json, logging, argparse
from pathlib import Path

import numpy as np
import torch
import h5py
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "CLAM"))
from models.model_clam import CLAM_SB, CLAM_MB
from wsi_core.WholeSlideImage import WholeSlideImage
from utils.file_utils import save_hdf5

# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="CLAM pipeline for Raspberry Pi")
    p.add_argument("--input",      required=True)
    p.add_argument("--output",     required=True)
    p.add_argument("--batch",      action="store_true", help="Batch mode: --input is a directory")
    p.add_argument("--ext",        default=".tif")
    p.add_argument("--ckpt",       default="checkpoints/s_0_checkpoint.pt")
    p.add_argument("--patch_size", type=int,   default=256)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--alpha",      type=float, default=0.4)
    p.add_argument("--use_mxa",    action="store_true", help="Use Memryx MXA for feature extraction")
    p.add_argument("--dfp",        default="resnet50_trunc.dfp", help="DFP file (only with --use_mxa)")
    p.add_argument("--tiff",       action="store_true", help="Also save full-res TIFF heatmap")
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

# ── MXA Feature Extractor ─────────────────────────────────────────────────────
class MXAResNet50:
    """Wraps Memryx MXA inference for ResNet50 feature extraction."""
    def __init__(self, dfp_path: str):
        import memryx as mx
        self.accl = mx.AsyncAccl(dfp_path)
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize(224),   # no CenterCrop — matches CPU path exactly
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    def extract_batch(self, pil_images: list) -> np.ndarray:
        """Extract features from a list of PIL images. Returns [N, 1024]."""
        tensors = torch.stack([self.transform(img) for img in pil_images])  # [N,3,224,224]
        results = []
        input_queue = []

        for i in range(len(tensors)):
            # Memryx expects float32 [1, C, H, W]
            inp = tensors[i].numpy().astype(np.float32)[np.newaxis, ...]  # [1,3,224,224]
            input_queue.append(inp)

        idx = [0]

        def send_input():
            if idx[0] < len(input_queue):
                inp = input_queue[idx[0]]
                idx[0] += 1
                return inp
            return None

        def collect_output(*outputs):
            feat = outputs[0].astype(np.float32).copy()  # force float32 after dequant
            # DFP outputs spatial feature maps [1, 1024, 14, 14] — apply GAP
            if feat.ndim == 4:
                feat = feat.mean(axis=(2, 3))  # → [1, 1024]
            elif feat.ndim == 3:
                feat = feat.mean(axis=(1, 2))  # → [1024]
            feat = feat.reshape(-1)
            # L2 normalize — corrects for int8 dequant scale drift,
            # makes features scale-invariant before CLAM attention
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            results.append(feat)

        self.accl.connect_input(send_input, model_idx=0)
        self.accl.connect_output(collect_output, model_idx=0)
        self.accl.wait()
        return np.vstack(results)  # [N, 1024]

# ── PyTorch Feature Extractor (CPU) ──────────────────────────────────────────
class PyTorchResNet50:
    """
    PyTorch ResNet50 feature extractor — matches server behavior exactly.
    Uses TimmCNNEncoder(features_only=True, out_indices=(3,)) + AdaptiveAvgPool2d(1)
    with transform: Resize(224) → ToTensor() → Normalize(ImageNet)
    """
    LOCAL_WEIGHTS = "checkpoints/resnet50_timm.pth"

    def __init__(self):
        import timm
        from torchvision import transforms
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"  # suppress HF Hub warning

        self.device = torch.device("cpu")
        # Exact same architecture as server's TimmCNNEncoder
        self.model = timm.create_model(
            "resnet50",
            features_only=True,
            out_indices=(3,),   # layer3 output → [N, 1024, H, W]
            pretrained=False,
            num_classes=0,
        )
        # Load local cached weights
        state = torch.load(self.LOCAL_WEIGHTS, map_location="cpu")
        # timm features_only model has different keys — load what matches
        self.model.load_state_dict(state, strict=False)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.model.eval()

        # Exact same transform as server (no CenterCrop)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_batch(self, pil_images: list) -> np.ndarray:
        """Extract features from a list of PIL images. Returns [N, 1024]."""
        tensors = torch.stack([self.transform(img) for img in pil_images])
        with torch.no_grad():
            out = self.model(tensors)   # list of feature maps; out_indices=(3,) → [N, 1024, H, W]
            features = self.pool(out[0]).squeeze(-1).squeeze(-1)  # [N, 1024]
        feats = features.cpu().numpy().astype(np.float32)
        # L2 normalize — keeps both paths consistent with MXA path
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return feats / norms

# ── Step 1: Patch Extraction ──────────────────────────────────────────────────
def step1(slide_path: Path, out: Path, patch_size: int, logger) -> dict:
    import subprocess
    t0 = time.time()
    logger.info("STEP 1 — Tissue Segmentation & Patch Extraction")
    slide_name = slide_path.stem
    step1_dir  = out / "step1_patches"
    for d in ["patches", "masks", "stitches"]:
        (step1_dir / d).mkdir(parents=True, exist_ok=True)

    # Use auto-generated CSV like the server does
    process_csv = step1_dir / "process_list_autogen.csv"
    with open(process_csv, "w") as f:
        f.write("slide_id,process\n")
        f.write(f"{slide_name}{slide_path.suffix},1\n")

    clam_dir = Path(__file__).parent / "CLAM"
    cmd = (f'python {clam_dir}/create_patches_fp.py '
           f'--source "{slide_path.parent.resolve()}" '
           f'--save_dir "{step1_dir.resolve()}" '
           f'--patch_size {patch_size} --step_size {patch_size} '
           f'--seg --patch --stitch '
           f'--process_list "{process_csv.resolve()}"')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Step 1 failed:\n{r.stderr}")

    patch_h5 = step1_dir / "patches" / f"{slide_name}.h5"
    with h5py.File(patch_h5, "r") as f:
        n_patches = len(f["coords"])
    elapsed = time.time() - t0
    logger.info(f"Patches extracted: {n_patches}  ({elapsed:.2f}s)")
    return {"n_patches": n_patches, "patch_h5": str(patch_h5),
            "process_csv": str(process_csv), "timing_seconds": round(elapsed, 3)}

# ── Step 2: Feature Extraction ───────────────────────────────────────────────
def step2(slide_path: Path, out: Path, extractor, patch_size: int, 
          batch_size: int, logger, use_mxa: bool = False) -> dict:
    import openslide
    t0 = time.time()
    backend = "MXA" if use_mxa else "PyTorch CPU"
    logger.info(f"STEP 2 — Feature Extraction (ResNet50 on {backend})")

    slide_name = slide_path.stem
    step1_dir  = out / "step1_patches"
    step2_dir  = out / "step2_features"
    (step2_dir / "pt_files").mkdir(parents=True, exist_ok=True)
    (step2_dir / "h5_files").mkdir(parents=True, exist_ok=True)

    patch_h5 = step1_dir / "patches" / f"{slide_name}.h5"
    with h5py.File(patch_h5, "r") as f:
        coords = f["coords"][:]

    try:
        wsi = openslide.open_slide(str(slide_path))
    except Exception:
        from wsi_core.tiff_slide import TiffSlide
        wsi = TiffSlide(str(slide_path))

    all_features = []
    n = len(coords)
    for start in range(0, n, batch_size):
        batch_coords = coords[start:start + batch_size]
        imgs = []
        for x, y in batch_coords:
            region = wsi.read_region((int(x), int(y)), 0, (patch_size, patch_size)).convert("RGB")
            imgs.append(region)
        feats = extractor.extract_batch(imgs)  # [B, 1024]
        all_features.append(feats)
        if (start // batch_size) % 5 == 0:
            logger.info(f"  {min(start + batch_size, n)}/{n} patches processed")

    features = np.vstack(all_features).astype(np.float32)  # [N, 1024]

    # save .pt and .h5
    pt_path = step2_dir / "pt_files" / f"{slide_name}.pt"
    torch.save(torch.from_numpy(features), str(pt_path))

    h5_path = step2_dir / "h5_files" / f"{slide_name}.h5"
    save_hdf5(str(h5_path), {"features": features, "coords": coords}, mode="w")

    elapsed = time.time() - t0
    logger.info(f"Features: {list(features.shape)}  ({elapsed:.2f}s)")
    return {"pt_file": str(pt_path), "feature_shape": list(features.shape),
            "timing_seconds": round(elapsed, 3)}

# ── Step 3: Inference + Heatmap ───────────────────────────────────────────────
def step3(slide_path: Path, out: Path, ckpt_path: Path,
          patch_size: int, alpha: float, logger, save_tiff: bool = False) -> dict:
    from scipy.stats import percentileofscore
    import openslide
    t0 = time.time()
    logger.info("STEP 3 — CLAM Inference + Heatmap Generation")

    slide_name = slide_path.stem
    step2_dir  = out / "step2_features"
    step3_dir  = out / "step3_heatmaps"
    step3_dir.mkdir(parents=True, exist_ok=True)

    # load features
    pt_path  = step2_dir / "pt_files" / f"{slide_name}.pt"
    features = torch.load(str(pt_path), map_location="cpu")

    # load CLAM model — auto-detect SB vs MB
    state = torch.load(str(ckpt_path), map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in state.items()
             if not k.startswith("instance_loss_fn")}
    is_mb = ("classifiers.0.weight" in state and "classifiers.1.weight" in state)
    ModelClass = CLAM_MB if is_mb else CLAM_SB
    logger.info(f"Model: {'CLAM_MB' if is_mb else 'CLAM_SB'}")
    model = ModelClass(gate=True, size_arg="small", dropout=0.25, n_classes=2, embed_dim=1024)
    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.no_grad():
        logits, Y_prob, Y_hat, A_raw, _ = model(features)

    Y_hat_val = Y_hat.item()
    probs     = Y_prob.squeeze().cpu().numpy()
    attn_raw  = A_raw.squeeze().cpu().numpy()
    attn      = attn_raw[Y_hat_val] if attn_raw.ndim == 2 else attn_raw
    label_map = {0: "Normal", 1: "Tumor"}
    logger.info(f"Prediction: {label_map[Y_hat_val]}  Normal={probs[0]:.4f}  Tumor={probs[1]:.4f}")

    with open(step3_dir / "prediction.txt", "w") as f:
        f.write(f"Slide     : {slide_name}\n")
        f.write(f"Prediction: {label_map[Y_hat_val]} (class {Y_hat_val})\n")
        f.write(f"Normal    : {probs[0]:.6f}\n")
        f.write(f"Tumor     : {probs[1]:.6f}\n")

    h5_path = step2_dir / "h5_files" / f"{slide_name}.h5"
    with h5py.File(str(h5_path), "r") as f:
        coords = f["coords"][:]

    attn_pct = np.array([percentileofscore(attn, s) for s in attn])

    # ── Variant 2: contrast-stretched scores (histogram equalization)
    # Spreads the attention distribution across the full [0,100] range,
    # restoring red/blue separation lost due to int8 quantization flattening.
    attn_sorted = np.sort(attn)
    attn_eq = np.array([percentileofscore(attn_sorted, s, kind="mean") for s in attn])

    # ── Variant 3: same contrast stretch but mapped to RdBu_r
    # RdBu_r: blue=low attention, red=high attention — perceptually cleaner
    # than jet for diverging attention scores.
    attn_eq_rdbu = attn_eq.copy()

    wsi_obj = WholeSlideImage(str(slide_path))
    try:
        _wsi = openslide.open_slide(str(slide_path))
        seg_level = _wsi.get_best_level_for_downsample(64)
        _wsi.close()
        logger.info(f"Backend: openslide  seg_level={seg_level}")
    except Exception:
        seg_level = 0
        logger.info("Backend: TiffSlide  vis_level=0 (full resolution forced)")

    wsi_obj.segmentTissue(
        seg_level=seg_level, sthresh=8, mthresh=7, close=4, use_otsu=False,
        filter_params={"a_t": 100, "a_h": 16, "max_n_holes": 8}
    )

    # ── Variant 1: original behavior (percentile + jet) ──────────────────────
    heatmap_v1 = wsi_obj.visHeatmap(
        scores=attn_pct, coords=coords, vis_level=0,
        patch_size=(patch_size, patch_size), alpha=alpha,
        blur=True, convert_to_percentiles=False,
        cmap="jet", blank_canvas=False, segment=True,
    )
    jpg_v1 = step3_dir / f"{slide_name}_heatmap_original.jpg"
    heatmap_v1.save(str(jpg_v1))
    logger.info(f"[V1 original]          → {jpg_v1}")

    # ── Variant 2: contrast-stretched + jet ──────────────────────────────────
    heatmap_v2 = wsi_obj.visHeatmap(
        scores=attn_eq, coords=coords, vis_level=0,
        patch_size=(patch_size, patch_size), alpha=alpha,
        blur=True, convert_to_percentiles=False,
        cmap="jet", blank_canvas=False, segment=True,
    )
    jpg_v2 = step3_dir / f"{slide_name}_heatmap_contrast_stretch.jpg"
    heatmap_v2.save(str(jpg_v2))
    logger.info(f"[V2 contrast+jet]      → {jpg_v2}")

    # ── Variant 3: contrast-stretched + RdBu_r ───────────────────────────────
    heatmap_v3 = wsi_obj.visHeatmap(
        scores=attn_eq_rdbu, coords=coords, vis_level=0,
        patch_size=(patch_size, patch_size), alpha=alpha,
        blur=True, convert_to_percentiles=False,
        cmap="RdBu_r", blank_canvas=False, segment=True,
    )
    jpg_v3 = step3_dir / f"{slide_name}_heatmap_recalibrated.jpg"
    heatmap_v3.save(str(jpg_v3))
    logger.info(f"[V3 contrast+RdBu_r]   → {jpg_v3}")

    # keep primary heatmap pointing to v1 for backward compat
    jpg_path = jpg_v1

    # always save full-res TIFF (matches server behavior)
    import tifffile as _tifffile
    tiff_path = step3_dir / f"{slide_name}_heatmap_original.tiff"
    _tifffile.imwrite(str(tiff_path), np.array(heatmap_v1),
                      photometric="rgb", compression="deflate",
                      metadata={"axes": "YXS"})
    logger.info(f"Saved TIFF → {tiff_path}  size={heatmap_v1.size}")

    wsi_raw  = wsi_obj.getOpenSlide()
    thumb_sz = wsi_raw.level_dimensions[0]
    orig = wsi_raw.read_region((0, 0), 0, thumb_sz).convert("RGB")
    orig.save(str(step3_dir / f"{slide_name}_original.jpg"))

    top_dir = step3_dir / "top_patches"
    top_dir.mkdir(exist_ok=True)
    for rank, idx in enumerate(np.argsort(attn)[-15:][::-1]):
        x, y = int(coords[idx][0]), int(coords[idx][1])
        wsi_raw.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB").save(
            str(top_dir / f"{rank:02d}_x{x}_y{y}_s{attn[idx]:.4f}.png"))

    elapsed = time.time() - t0
    logger.info(f"Step 3 done in {elapsed:.2f}s")
    logger.info("Heatmap variants saved:")
    logger.info(f"  V1 original (percentile+jet)      : {jpg_v1.name}")
    logger.info(f"  V2 contrast stretch (eq+jet)       : {jpg_v2.name}")
    logger.info(f"  V3 recalibrated (eq+RdBu_r)        : {jpg_v3.name}")
    return {"prediction": label_map[Y_hat_val], "prob_normal": float(probs[0]),
            "prob_tumor": float(probs[1]),
            "heatmap_v1_original": str(jpg_v1),
            "heatmap_v2_contrast_stretch": str(jpg_v2),
            "heatmap_v3_recalibrated": str(jpg_v3),
            "heatmap_tiff": str(tiff_path),
            "timing_seconds": round(elapsed, 3)}

# ── Run single slide ──────────────────────────────────────────────────────────
def run_slide(slide_path: Path, output_dir: Path, args, extractor):
    out = output_dir / slide_path.stem
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out / f"{slide_path.stem}.log")
    logger.info(f"Input  : {slide_path}")
    logger.info(f"Output : {out}")
    t0 = time.time()
    status = "SUCCESS"
    try:
        step1(slide_path, out, args.patch_size, logger)
        step2(slide_path, out, extractor, args.patch_size, args.batch_size, logger, use_mxa=args.use_mxa)
        step3(slide_path, out, Path(args.ckpt), args.patch_size, args.alpha, logger, save_tiff=args.tiff)
    except Exception as e:
        status = f"FAILED: {e}"
        logger.error(f"Pipeline FAILED: {e}")
    elapsed = round(time.time() - t0, 2)
    summary = {"slide": str(slide_path), "status": status, "total_time_seconds": elapsed}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Status: {status}  Total: {elapsed}s")
    return summary

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Choose feature extractor
    if args.use_mxa:
        if not Path(args.dfp).exists():
            print(f"[ERROR] DFP not found: {args.dfp}")
            print("  Run compile_resnet.sh first to generate the DFP from resnet50_trunc.onnx")
            sys.exit(1)
        print(f"Loading MXA with DFP: {args.dfp}")
        extractor = MXAResNet50(args.dfp)
    else:
        print("Loading PyTorch ResNet50 (CPU) — matches server behavior")
        extractor = PyTorchResNet50()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        slides = sorted(Path(args.input).glob(f"*{args.ext}"))
        if not slides:
            print(f"[ERROR] No {args.ext} files found in {args.input}")
            sys.exit(1)
        print(f"Found {len(slides)} slide(s)")
        summaries = []
        t0 = time.time()
        for slide in slides:
            print(f"\n{'='*60}\nProcessing: {slide.name}\n{'='*60}")
            summaries.append(run_slide(slide, output_dir, args, extractor))
        with open(output_dir / "all_slides_summary.json", "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nAll {len(slides)} slides done in {round(time.time()-t0, 2)}s")
    else:
        slide_path = Path(args.input)
        if not slide_path.exists():
            print(f"[ERROR] Slide not found: {slide_path}")
            sys.exit(1)
        run_slide(slide_path, output_dir, args, extractor)

if __name__ == "__main__":
    main()
