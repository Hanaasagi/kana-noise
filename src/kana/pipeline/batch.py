import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from .steps import (
    ExtractStep,
    DemucsStep,
    VADStep,
    ScoreStep,
    IntersectStep,
    PannsStep,
    ParalinguisticStep,
    ExportStep,
)
from .utils import ensure_dir

logger = logging.getLogger(__name__)


def video_stem(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return stem


def process_one(
    video_path: str,
    runs_root: str,
    reuse: bool = True,
    preview_minutes: float = 0.0,
    # demucs
    separate_vocals: bool = False,
    demucs_model: str = "mdx_q",
    demucs_device: str = "cpu",
    demucs_jobs: int = 0,
    demucs_two_stems: str = "vocals",
    demucs_only: bool = False,
    # vad
    vad_aggr: int = 2,
    vad_min_ms: int = 150,
    vad_merge_ms: int = 200,
    # score
    score_thr: float = 0.40,
    min_sec: float = 0.20,
    merge_gap: float = 0.30,
    # intersect
    pad_sec: float = 0.20,
    # panns
    use_panns: bool = False,
    panns_thr: float = 0.25,
    anti_singing: bool = True,
    # paralinguistic
    extract_quirks: bool = False,
    quirks_panns: bool = False,
    quirks_panns_thr: float = 0.30,
    # export
    reencode: bool = False,
    do_export: bool = True,
) -> str:
    stem = video_stem(video_path)
    run_dir = os.path.join(runs_root, stem)
    ensure_dir(run_dir)

    extract = ExtractStep(
        run_dir,
        video_path,
        sr16=16000,
        sr44=44100,
        preview_seconds=preview_minutes * 60.0,
    )
    extract.execute(reuse=reuse)

    paths = extract.paths()
    if separate_vocals:
        demucs = DemucsStep(
            run_dir,
            wav44=paths["wav44"],
            model=demucs_model,
            device=demucs_device,
            jobs=demucs_jobs,
            two_stems=demucs_two_stems,
        )
        demucs.execute(reuse=reuse)
        if demucs_only:
            return run_dir

    vad = VADStep(
        run_dir, aggressiveness=vad_aggr, min_ms=vad_min_ms, merge_ms=vad_merge_ms
    )
    vad.execute(reuse=reuse)

    score = ScoreStep(run_dir, thr=score_thr, min_sec=min_sec, merge_gap=merge_gap)
    score.execute(reuse=reuse)

    inter = IntersectStep(run_dir, pad_sec=pad_sec)
    inter.execute(reuse=reuse)

    panns = PannsStep(
        run_dir, use_panns=use_panns, panns_thr=panns_thr, anti_singing=anti_singing
    )
    panns.execute(reuse=reuse)

    quirks = ParalinguisticStep(
        run_dir,
        enable=extract_quirks,
        panns_enable=quirks_panns,
        panns_thr=quirks_panns_thr,
    )
    quirks.execute(reuse=reuse)

    if do_export:
        export = ExportStep(run_dir, video_path=video_path, reencode=reencode)
        export.execute(reuse=reuse)

    return run_dir


def batch_process(
    videos_dir: str,
    runs_root: str,
    patterns: Optional[List[str]] = None,
    file_list: Optional[List[str]] = None,
    num_workers: int = 2,
    **kwargs,
) -> List[str]:
    ensure_dir(runs_root)
    # Collect mp4 files
    if file_list is not None and len(file_list) > 0:
        files = file_list
    else:
        files = [
            os.path.join(videos_dir, f)
            for f in os.listdir(videos_dir)
            if f.lower().endswith(".mp4")
        ]
    files.sort()
    results = []
    logger.info("Batch start: %d files", len(files))
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        futs = {ex.submit(process_one, f, runs_root, **kwargs): f for f in files}
        for fut in as_completed(futs):
            try:
                run_dir = fut.result()
                results.append(run_dir)
                logger.info("Done: %s", futs[fut])
            except Exception as e:
                logger.exception("Failed on %s: %s", futs[fut], e)
    logger.info("Batch finished: %d ok", len(results))
    return results
