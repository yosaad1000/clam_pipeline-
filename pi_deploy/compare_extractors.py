"""
compare_extractors.py — Run both PyTorchResNet50 and MXAResNet50 on the same
slide, compare the feature vectors they produce, and report differences before
the features would be fed into Step 3 (CLAM inference).

Usage:
    python compare_extractors.py --input slide.tif --dfp resnet50_trunc.dfp
    python compare_extractors.py --input slide.tif --dfp resnet50_trunc.dfp --output out_dir

Output (inside --output/<slide_name>/):
    pytorch_features.pt          — [N, 1024] float32 from PyTorchResNet50
    mxa_features.pt              — [N, 1024] float32 from MXAResNet50
    diff_report.json             — per-patch and aggregate difference stats
    diff_histogram.png           — histogram of per-patch cosine similarities
"""
import os, sys, time, argparse, json
from pathlib import Path

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── resolve paths so we can import from pipeline_mxa and CLAM ────────────────
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "CLAM"))

from pipeline_mxa import (
    PyTorchResNet50,
    MXAResNet50,
    step1,
    setup_logger,
)
from utils.file_utils import save_hdf5


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Compare PyTorch vs MXA feature extractors")
    p.add_argument("--input",      required=True,  help="Path to a single WSI file")
    p.add_argument("--dfp",        required=True,  help="Memryx DFP file for ResNet50")
    p.add_argument("--output",     default="compare_output", help="Output directory")
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def extract_features(slide_path: Path, coords, extractor, patch_size: int,
                     batch_size: int, label: str, logger) -> np.ndarray:
    """Run extractor over all patches and return [N, 1024] float32 array."""
    try:
        import openslide
        wsi = openslide.open_slide(str(slide_path))
    except Exception:
        from wsi_core.tiff_slide import TiffSlide
        wsi = TiffSlide(str(slide_path))

    all_features = []
    n = len(coords)
    t0 = time.time()

    for start in range(0, n, batch_size):
        batch_coords = coords[start:start + batch_size]
        imgs = [
            wsi.read_region((int(x), int(y)), 0, (patch_size, patch_size)).convert("RGB")
            for x, y in batch_coords
        ]
        feats = extractor.extract_batch(imgs)   # [B, 1024]
        all_features.append(feats)

        done = min(start + batch_size, n)
        if (start // batch_size) % 5 == 0:
            logger.info(f"  [{label}] {done}/{n} patches")

    elapsed = time.time() - t0
    features = np.vstack(all_features).astype(np.float32)
    logger.info(f"  [{label}] done — shape={list(features.shape)}  time={elapsed:.2f}s")
    return features


# ─────────────────────────────────────────────────────────────────────────────
def compute_diff(pt_feats: np.ndarray, mxa_feats: np.ndarray) -> dict:
    """
    Compute per-patch and aggregate difference metrics between two [N, 1024]
    feature matrices.

    Metrics:
      - cosine_similarity   : dot product of L2-normalised rows (both already
                              normalised, so this is just the dot product)
      - l2_distance         : Euclidean distance per patch
      - l1_distance         : Mean absolute difference per patch
      - max_abs_diff        : Max absolute element-wise diff per patch
    """
    N = pt_feats.shape[0]

    # cosine similarity — features are already L2-normalised by both extractors
    cos_sim = np.einsum("ij,ij->i", pt_feats, mxa_feats)          # [N]

    diff    = pt_feats - mxa_feats                                  # [N, 1024]
    l2_dist = np.linalg.norm(diff, axis=1)                         # [N]
    l1_dist = np.abs(diff).mean(axis=1)                            # [N]
    max_abs = np.abs(diff).max(axis=1)                             # [N]

    def stats(arr):
        return {
            "mean":   float(arr.mean()),
            "std":    float(arr.std()),
            "min":    float(arr.min()),
            "max":    float(arr.max()),
            "median": float(np.median(arr)),
            "p95":    float(np.percentile(arr, 95)),
        }

    return {
        "n_patches": N,
        "feature_dim": pt_feats.shape[1],
        "cosine_similarity": stats(cos_sim),
        "l2_distance":       stats(l2_dist),
        "l1_distance":       stats(l1_dist),
        "max_abs_diff":      stats(max_abs),
        # per-patch arrays (saved separately for inspection)
        "_per_patch": {
            "cosine_similarity": cos_sim.tolist(),
            "l2_distance":       l2_dist.tolist(),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
def save_histogram(cos_sim: list, l2_dist: list, out_path: Path):
    cos_sim = np.array(cos_sim)
    l2_dist = np.array(l2_dist)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(cos_sim, bins=50, color="steelblue", edgecolor="white")
    axes[0].axvline(cos_sim.mean(), color="red", linestyle="--",
                    label=f"mean={cos_sim.mean():.4f}")
    axes[0].set_title("Cosine Similarity (PyTorch vs MXA)")
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].set_ylabel("Patch count")
    axes[0].legend()

    axes[1].hist(l2_dist, bins=50, color="darkorange", edgecolor="white")
    axes[1].axvline(l2_dist.mean(), color="red", linestyle="--",
                    label=f"mean={l2_dist.mean():.4f}")
    axes[1].set_title("L2 Distance (PyTorch vs MXA)")
    axes[1].set_xlabel("L2 Distance")
    axes[1].set_ylabel("Patch count")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
def print_summary(report: dict, logger):
    logger.info("=" * 60)
    logger.info("FEATURE COMPARISON SUMMARY")
    logger.info(f"  Patches     : {report['n_patches']}")
    logger.info(f"  Feature dim : {report['feature_dim']}")
    logger.info("")

    cs = report["cosine_similarity"]
    logger.info("  Cosine Similarity (1.0 = identical direction)")
    logger.info(f"    mean={cs['mean']:.6f}  std={cs['std']:.6f}  "
                f"min={cs['min']:.6f}  p95={cs['p95']:.6f}")

    l2 = report["l2_distance"]
    logger.info("  L2 Distance (0.0 = identical)")
    logger.info(f"    mean={l2['mean']:.6f}  std={l2['std']:.6f}  "
                f"max={l2['max']:.6f}  p95={l2['p95']:.6f}")

    l1 = report["l1_distance"]
    logger.info("  L1 Distance (mean abs element diff)")
    logger.info(f"    mean={l1['mean']:.6f}  std={l1['std']:.6f}")

    # quick verdict
    if cs["mean"] > 0.99:
        verdict = "EXCELLENT — features are nearly identical"
    elif cs["mean"] > 0.95:
        verdict = "GOOD — minor quantisation drift, unlikely to affect CLAM"
    elif cs["mean"] > 0.90:
        verdict = "MODERATE — noticeable drift, worth investigating"
    else:
        verdict = "POOR — significant divergence, check DFP calibration"

    logger.info("")
    logger.info(f"  Verdict: {verdict}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    slide_path = Path(args.input).resolve()
    out_root   = Path(args.output) / slide_path.stem
    out_root.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_root / "compare.log")
    logger.info(f"Slide  : {slide_path}")
    logger.info(f"DFP    : {args.dfp}")
    logger.info(f"Output : {out_root}")

    # ── Step 1: patch extraction (shared between both extractors) ─────────────
    logger.info("\n--- Step 1: Patch Extraction ---")
    step1_result = step1(slide_path, out_root, args.patch_size, logger)
    patch_h5 = Path(step1_result["patch_h5"])

    with h5py.File(patch_h5, "r") as f:
        coords = f["coords"][:]
    logger.info(f"Loaded {len(coords)} patch coordinates from {patch_h5.name}")

    # ── Step 2a: PyTorch CPU extraction ──────────────────────────────────────
    logger.info("\n--- Step 2a: PyTorch CPU Feature Extraction ---")
    pytorch_extractor = PyTorchResNet50()
    pt_feats = extract_features(
        slide_path, coords, pytorch_extractor,
        args.patch_size, args.batch_size, "PyTorch", logger
    )
    pt_out = out_root / "pytorch_features.pt"
    torch.save(torch.from_numpy(pt_feats), str(pt_out))
    logger.info(f"Saved → {pt_out}")

    # ── Step 2b: MXA extraction ───────────────────────────────────────────────
    logger.info("\n--- Step 2b: MXA Feature Extraction ---")
    mxa_extractor = MXAResNet50(args.dfp)
    mxa_feats = extract_features(
        slide_path, coords, mxa_extractor,
        args.patch_size, args.batch_size, "MXA", logger
    )
    mxa_out = out_root / "mxa_features.pt"
    torch.save(torch.from_numpy(mxa_feats), str(mxa_out))
    logger.info(f"Saved → {mxa_out}")

    # ── Diff ──────────────────────────────────────────────────────────────────
    logger.info("\n--- Computing Differences ---")
    report = compute_diff(pt_feats, mxa_feats)

    # strip per-patch arrays before saving the clean JSON report
    per_patch = report.pop("_per_patch")

    report_path = out_root / "diff_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved diff report → {report_path}")

    hist_path = out_root / "diff_histogram.png"
    save_histogram(per_patch["cosine_similarity"], per_patch["l2_distance"], hist_path)
    logger.info(f"Saved histogram   → {hist_path}")

    print_summary(report, logger)

    logger.info("\nDone. Features ready for Step 3.")
    logger.info(f"  PyTorch features : {pt_out}")
    logger.info(f"  MXA features     : {mxa_out}")


if __name__ == "__main__":
    main()
