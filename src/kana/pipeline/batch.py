import logging
from pathlib import Path
from typing import List, Optional, Union

from .steps import (
    ExtractStep,
    DemucsStep,
    VADStep,
    ScoreStep,
    IntersectStep,
    PannsStep,
    ParalinguisticStep,
    ExportStep,
    SubtitleStep,
)
from .utils import ensure_dir

logger = logging.getLogger(__name__)


PathLike = Union[str, Path]


def video_stem(path: str) -> str:
    p = Path(path)
    return p.stem


def process_one(
    video_path: PathLike,
    runs_root: PathLike,
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
    # subtitle
    gen_subs: bool = False,
    whisper_model: Optional[str] = None,
    subs_language: str = "auto",
    # export
    reencode: bool = False,
    do_export: bool = True,
) -> str:
    video_path = Path(video_path)
    runs_root = Path(runs_root)
    stem = video_stem(str(video_path))
    run_dir = runs_root / stem
    ensure_dir(run_dir)

    extract = ExtractStep(
        str(run_dir),
        str(video_path),
        sr16=16000,
        sr44=44100,
        preview_seconds=preview_minutes * 60.0,
    )
    extract.execute(reuse=reuse)

    paths = extract.paths()
    if separate_vocals:
        demucs = DemucsStep(
            str(run_dir),
            wav44=paths["wav44"],
            model=demucs_model,
            device=demucs_device,
            jobs=demucs_jobs,
            two_stems=demucs_two_stems,
        )
        demucs.execute(reuse=reuse)
        if demucs_only:
            return str(run_dir)

    vad = VADStep(
        str(run_dir), aggressiveness=vad_aggr, min_ms=vad_min_ms, merge_ms=vad_merge_ms
    )
    vad.execute(reuse=reuse)

    score = ScoreStep(str(run_dir), thr=score_thr, min_sec=min_sec, merge_gap=merge_gap)
    score.execute(reuse=reuse)

    inter = IntersectStep(str(run_dir), pad_sec=pad_sec)
    inter.execute(reuse=reuse)

    panns = PannsStep(
        str(run_dir),
        use_panns=use_panns,
        panns_thr=panns_thr,
        anti_singing=anti_singing,
    )
    panns.execute(reuse=reuse)

    quirks = ParalinguisticStep(
        str(run_dir),
        enable=extract_quirks,
        panns_enable=quirks_panns,
        panns_thr=quirks_panns_thr,
    )
    quirks.execute(reuse=reuse)

    subs = SubtitleStep(
        str(run_dir), enable=gen_subs, model_path=whisper_model, language=subs_language
    )
    subs.execute(reuse=reuse)

    if do_export:
        export = ExportStep(str(run_dir), video_path=str(video_path), reencode=reencode)
        export.execute(reuse=reuse)

    return str(run_dir)


def batch_process(
    videos_dir: PathLike,
    runs_root: PathLike,
    patterns: Optional[List[str]] = None,
    file_list: Optional[List[str]] = None,
    **kwargs,
) -> List[str]:
    runs_root = Path(runs_root)
    videos_dir = Path(videos_dir)
    ensure_dir(runs_root)
    # Collect mp4 files
    if file_list is not None and len(file_list) > 0:
        files = [str(Path(p)) for p in file_list]
    else:
        files = [str(p) for p in videos_dir.iterdir() if p.suffix.lower() == ".mp4"]
    files.sort()
    results: List[str] = []
    logger.info("Batch start: %d files", len(files))
    for f in files:
        try:
            run_dir = process_one(f, runs_root, **kwargs)
            results.append(run_dir)
            logger.info("Done: %s", f)
        except Exception as e:
            logger.exception("Failed on %s: %s", f, e)
    logger.info("Batch finished: %d ok", len(results))
    return results
