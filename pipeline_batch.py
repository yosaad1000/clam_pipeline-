"""
pipeline_batch.py — CLAM pipeline for a directory of WSI images.

Usage:
    conda run -n clam python pipeline_batch.py \
        --input  path/to/slides_dir \
        --output path/to/output_dir \
        [--ext .tif] \
        [--patch_size 256] \
        [--batch_size 256] \
        [--alpha 0.4] \
        [--ckpt CLAM/heatmaps/demo/ckpts/s_0_checkpoint.pt]
"""
import sys, time, json, logging, traceback, argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "CLAM")

# reuse all step functions from pipeline_single
from pipeline_single import step1, step2, step3, setup_logger, save_meta

# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="CLAM batch pipeline")
    p.add_argument("--input",      required=True,  help="Directory containing WSI files")
    p.add_argument("--output",     required=True,  help="Output root directory")
    p.add_argument("--ext",        default=".tif", help="Slide file extension (default: .tif)")
    p.add_argument("--ckpt",       default="CLAM/heatmaps/demo/ckpts/s_0_checkpoint.pt")
    p.add_argument("--patch_size", type=int,   default=256)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--alpha",      type=float, default=0.4)
    p.add_argument("--model_name", type=str,   default="resnet50_trunc",
                   choices=["resnet50_trunc", "uni_v1", "conch_v1"],
                   help="Feature extractor model")
    p.add_argument("--uni_ckpt",   type=str,   default=None,
                   help="Path to UNI weights (required when --model_name uni_v1)")
    p.add_argument("--conch_ckpt", type=str,   default=None,
                   help="Path to CONCH weights (required when --model_name conch_v1)")
    return p.parse_args()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    ckpt_path  = Path(args.ckpt)

    if not input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # discover slides
    ext = args.ext if args.ext.startswith(".") else f".{args.ext}"
    slides = sorted(input_dir.glob(f"*{ext}"))
    if not slides:
        print(f"[ERROR] No {ext} files found in {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nFound {len(slides)} slide(s): {[s.name for s in slides]}\n")

    grand_start = time.time()
    all_summaries = []

    for slide_path in slides:
        out = output_dir / slide_path.stem
        out.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(out / f"{slide_path.stem}.log")

        t_start = time.time()
        print(f"\n{'='*60}\nProcessing: {slide_path.name}\n{'='*60}")
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
        all_summaries.append(summary)
        logger.info(f"Status: {status}  Total: {total:.2f}s")

    grand_total = time.time() - grand_start
    global_meta = {
        "run_at": datetime.now().isoformat(),
        "total_slides": len(slides),
        "grand_total_seconds": round(grand_total, 3),
        "slides": all_summaries,
    }
    save_meta(output_dir / "all_slides_summary.json", global_meta)

    print(f"\n{'='*60}")
    print(f"All {len(slides)} slides done in {grand_total:.2f}s")
    print(f"Summary → {output_dir}/all_slides_summary.json")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
