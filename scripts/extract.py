#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys

from kana.pipeline.batch import batch_process


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch process videos to separate vocals and prepare for voice detection"
    )
    parser.add_argument(
        "-d",
        "--videos-dir",
        default="videos",
        help="Directory containing input .mp4 files",
    )
    parser.add_argument(
        "-r",
        "--runs-root",
        default="runs",
        help="Root directory to store per-video outputs",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse cached step outputs when signatures match",
    )
    parser.add_argument(
        "--no-reuse", action="store_true", help="Disable reuse (force recompute)"
    )
    parser.add_argument(
        "--preview-minutes",
        type=float,
        default=0.0,
        help="Process only first N minutes",
    )
    # removed: num-workers (GPU/MPS memory contention)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    # Demucs
    parser.add_argument(
        "--separate-vocals",
        action="store_true",
        help="Use Demucs separation before detection",
    )
    parser.add_argument("--demucs-model", default="mdx_q")
    parser.add_argument(
        "--demucs-device", default="cpu", choices=["cpu", "cuda", "mps"]
    )
    parser.add_argument("--demucs-jobs", type=int, default=0)
    parser.add_argument(
        "--demucs-two-stems",
        default="vocals",
        choices=["vocals", "drums", "bass", "other"],
    )
    parser.add_argument(
        "--demucs-only", action="store_true", help="Only run Demucs and exit"
    )
    # VAD
    parser.add_argument("--vad-aggr", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--vad-min-ms", type=int, default=150)
    parser.add_argument("--vad-merge-ms", type=int, default=200)
    # Score
    parser.add_argument("--score-thr", type=float, default=0.40)
    parser.add_argument("--min-sec", type=float, default=0.20)
    parser.add_argument("--merge-gap", type=float, default=0.30)
    # Intersect
    parser.add_argument("--pad-sec", type=float, default=0.20)
    # PANNs
    parser.add_argument("--use-panns", action="store_true")
    parser.add_argument("--panns-thr", type=float, default=0.25)
    parser.add_argument("--no-anti-singing", action="store_true")
    # Paralinguistic
    parser.add_argument(
        "--extract-quirks",
        action="store_true",
        help="Extract paralinguistic events (click/murmur/sigh)",
    )
    parser.add_argument(
        "--quirks-panns",
        action="store_true",
        help="Use PANNs classes for paralinguistic (hum/whisper/sigh/etc)",
    )
    parser.add_argument("--quirks-panns-thr", type=float, default=0.30)
    # Subtitle
    parser.add_argument(
        "--gen-subs",
        action="store_true",
        help="Generate subtitles with whisper.cpp after Demucs",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=None,
        help="Path to whisper.cpp .bin model (optional)",
    )
    parser.add_argument(
        "--subs-language",
        type=str,
        default="auto",
        help="Whisper language (auto/zh/ja/en/...)",
    )
    # Export
    parser.add_argument("--reencode", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    # Config
    parser.add_argument(
        "--config", type=str, help="Path to config.json (CLI overrides config)"
    )
    # Selection
    parser.add_argument(
        "--all", action="store_true", help="Process all videos under --videos-dir"
    )
    parser.add_argument(
        "--file",
        type=str,
        action="append",
        help="Process a specific video file (can repeat)",
    )
    return parser


def main():
    # Pre-parse config
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str)
    pre_args, _ = pre.parse_known_args()

    parser = build_parser()

    if pre_args.config:
        cfg_path = os.path.abspath(pre_args.config)
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if isinstance(cfg, dict):
                    parser.set_defaults(**cfg)
            except Exception as e:
                print(f"Failed to load config: {cfg_path}: {e}")
        else:
            print(f"Config not found: {cfg_path}")

    args = parser.parse_args()
    reuse = args.reuse and not args.no_reuse

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Interactive selection if neither --all nor --file provided
    file_list = None
    if not args.all and not args.file:
        vd = os.path.abspath(args.videos_dir)
        items = [f for f in os.listdir(vd) if f.lower().endswith(".mp4")]
        items.sort()
        if not items:
            print(f"No mp4 files under: {vd}")
            return 0
        print("Select videos to process (comma-separated indices), or 'a' for all:")
        for i, f in enumerate(items):
            print(f"[{i}] {f}")
        sel = input("> ").strip()
        if sel.lower() == "a":
            file_list = [os.path.join(vd, f) for f in items]
        else:
            try:
                idxs = [int(x) for x in sel.split(",") if x.strip().isdigit()]
                file_list = [
                    os.path.join(vd, items[i]) for i in idxs if 0 <= i < len(items)
                ]
            except Exception:
                print("Invalid selection.")
                return 1
    elif args.file:
        file_list = [os.path.abspath(p) for p in args.file]

    results = batch_process(
        videos_dir=args.videos_dir,
        runs_root=args.runs_root,
        reuse=reuse,
        preview_minutes=args.preview_minutes,
        separate_vocals=args.separate_vocals,
        demucs_model=args.demucs_model,
        demucs_device=args.demucs_device,
        demucs_jobs=args.demucs_jobs,
        demucs_two_stems=args.demucs_two_stems,
        demucs_only=args.demucs_only,
        vad_aggr=args.vad_aggr,
        vad_min_ms=args.vad_min_ms,
        vad_merge_ms=args.vad_merge_ms,
        score_thr=args.score_thr,
        min_sec=args.min_sec,
        merge_gap=args.merge_gap,
        pad_sec=args.pad_sec,
        use_panns=args.use_panns,
        panns_thr=args.panns_thr,
        anti_singing=(not args.no_anti_singing),
        extract_quirks=args.extract_quirks,
        quirks_panns=args.quirks_panns,
        quirks_panns_thr=args.quirks_panns_thr,
        gen_subs=args.gen_subs,
        whisper_model=args.whisper_model,
        subs_language=args.subs_language,
        reencode=args.reencode,
        do_export=(not args.no_export),
        file_list=file_list,
    )
    print("Processed:")
    for r in results:
        print("-", r)


if __name__ == "__main__":
    sys.exit(main())
