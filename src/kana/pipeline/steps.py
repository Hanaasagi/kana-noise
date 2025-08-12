import os
import subprocess
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import wave
import numpy as np
import librosa
import webrtcvad

try:
    from panns_inference import AudioTagging as _PannsAT  # type: ignore
    from panns_inference import labels as _PANN_LABELS  # type: ignore

    _HAS_PANNS = True
except Exception:
    _HAS_PANNS = False

from .utils import ensure_dir, file_info, read_json, write_json


logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    skipped: bool
    reason: str


class Step:
    name: str = "step"

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        ensure_dir(os.path.join(self.run_dir, "meta"))
        ensure_dir(os.path.join(self.run_dir, "logs"))

    def meta_path(self) -> str:
        return os.path.join(self.run_dir, "meta", f"{self.name}.json")

    def outputs_ready(self) -> bool:
        raise NotImplementedError

    def signature(self) -> Dict[str, Any]:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def execute(self, reuse: bool = True) -> StepResult:
        meta_fp = self.meta_path()
        sig = self.signature()
        if reuse:
            meta = read_json(meta_fp) or {}
            if meta.get("signature") == sig and self.outputs_ready():
                logger.info("[%s] Skip (cache hit)", self.name)
                return StepResult(skipped=True, reason="signature matched")

        # Run step
        logger.info("[%s] Start", self.name)
        self.run()
        # Write meta
        meta = {"signature": sig, "status": "ok"}
        write_json(meta_fp, meta)
        logger.info("[%s] Done", self.name)
        return StepResult(skipped=False, reason="executed")


class ExtractStep(Step):
    name = "extract"

    def __init__(
        self,
        run_dir: str,
        video_path: str,
        sr16: int = 16000,
        sr44: int = 44100,
        preview_seconds: float = 0.0,
    ):
        super().__init__(run_dir)
        self.video_path = video_path
        self.sr16 = sr16
        self.sr44 = sr44
        self.preview_seconds = preview_seconds

    def paths(self) -> Dict[str, str]:
        audio_dir = os.path.join(self.run_dir, "audio")
        ensure_dir(audio_dir)
        return {
            "wav16": os.path.join(audio_dir, "extracted_16k.wav"),
            "wav44": os.path.join(audio_dir, "extracted_44k.wav"),
        }

    def outputs_ready(self) -> bool:
        p = self.paths()
        return file_info(p["wav16"]).get("exists", False) and file_info(p["wav44"]).get(
            "exists", False
        )

    def signature(self) -> Dict[str, Any]:
        vinfo = file_info(self.video_path)
        return {
            "video": {"path": self.video_path, **vinfo},
            "sr16": self.sr16,
            "sr44": self.sr44,
            "preview": self.preview_seconds,
        }

    def run(self) -> None:
        p = self.paths()
        # 44.1k stereo
        logger.info("[extract] 44.1k stereo → %s", p["wav44"])
        cmd44 = [
            "ffmpeg",
            "-y",
            "-i",
            self.video_path,
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(self.sr44),
            "-sample_fmt",
            "s16",
        ]
        if self.preview_seconds and self.preview_seconds > 0:
            cmd44 += ["-t", f"{self.preview_seconds}"]
        cmd44 += [p["wav44"]]
        subprocess.run(
            cmd44, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # 16k mono
        logger.info("[extract] 16k mono → %s", p["wav16"])
        cmd16 = [
            "ffmpeg",
            "-y",
            "-i",
            self.video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.sr16),
            "-sample_fmt",
            "s16",
        ]
        if self.preview_seconds and self.preview_seconds > 0:
            cmd16 += ["-t", f"{self.preview_seconds}"]
        cmd16 += [p["wav16"]]
        subprocess.run(
            cmd16, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )


class DemucsStep(Step):
    name = "demucs"

    def __init__(
        self,
        run_dir: str,
        wav44: str,
        model: str = "mdx_q",
        device: str = "cpu",
        jobs: int = 0,
        two_stems: str = "vocals",
    ):
        super().__init__(run_dir)
        self.wav44 = wav44
        self.model = model
        self.device = device
        self.jobs = jobs
        self.two_stems = two_stems

    def paths(self) -> Dict[str, str]:
        audio_dir = os.path.join(self.run_dir, "audio")
        base = os.path.splitext(os.path.basename(self.wav44))[0]
        sep_dir = os.path.join(audio_dir, "demucs", self.model, base)
        ensure_dir(sep_dir)
        return {
            "sep_dir": sep_dir,
            "vocals": os.path.join(sep_dir, "vocals.wav"),
            "no_vocals": os.path.join(sep_dir, "no_vocals.wav"),
            "vocals16": os.path.join(audio_dir, "vocals_16k.wav"),
        }

    def outputs_ready(self) -> bool:
        p = self.paths()
        return file_info(p["vocals"]).get("exists", False) and file_info(
            p["vocals16"]
        ).get("exists", False)

    def signature(self) -> Dict[str, Any]:
        return {
            "wav44": file_info(self.wav44),
            "model": self.model,
            "jobs": self.jobs,
            "two_stems": self.two_stems,
        }

    def run(self) -> None:
        p = self.paths()
        out_root = os.path.dirname(os.path.dirname(p["sep_dir"]))
        cmd = ["demucs", "-n", self.model, "-o", out_root]
        if self.device in ("cpu", "cuda", "mps"):
            cmd += ["-d", self.device]
        if self.jobs and self.jobs > 0:
            cmd += ["-j", str(self.jobs)]
        if self.two_stems in ("vocals", "drums", "bass", "other"):
            cmd += ["--two-stems", self.two_stems]
        cmd += [self.wav44]
        logger.info("[demucs] cmd: %s", " ".join(cmd))
        log_path = os.path.join(self.run_dir, "logs", "demucs.log")
        with open(log_path, "w", encoding="utf-8", buffering=1) as lf:
            lf.write("# Demucs logs\n")
            lf.write(f"cmd: {' '.join(cmd)}\n\n")
            subprocess.run(cmd, check=True, stdout=lf, stderr=lf)
        logger.info("[demucs] log: %s", log_path)

        # Downmix vocals to 16k mono for detection
        logger.info("[demucs] vocals → %s", p["vocals"])
        logger.info("[demucs] downmix vocals to 16k → %s", p["vocals16"])
        cmd_down = [
            "ffmpeg",
            "-y",
            "-i",
            p["vocals"],
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            p["vocals16"],
        ]
        subprocess.run(
            cmd_down, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )


def _detect_wav(run_dir: str) -> str:
    """Prefer vocals_16k.wav else extracted_16k.wav."""
    cand = os.path.join(run_dir, "audio", "vocals_16k.wav")
    if os.path.isfile(cand):
        return cand
    return os.path.join(run_dir, "audio", "extracted_16k.wav")


# ========== VAD utilities ==========


def _read_wave_bytes(wav_path: str) -> Tuple[bytes, int]:
    with wave.open(wav_path, "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        if (
            num_channels != 1
            or sample_width != 2
            or sample_rate not in (8000, 16000, 32000, 48000)
        ):
            raise ValueError(
                f"webrtcvad requires mono 16-bit PCM at 8/16/32/48kHz. Got channels={num_channels}, width={sample_width}, sr={sample_rate}."
            )
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def _frame_generator(pcm: bytes, sample_rate: int, frame_duration_ms: int):
    bytes_per_sample = 2
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0)) * bytes_per_sample
    num_frames = len(pcm) // frame_size
    timestamp = 0.0
    duration = frame_duration_ms / 1000.0
    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        yield (timestamp, duration, pcm[start:end])
        timestamp += duration


def _collect_vad_segments(
    wav_path: str,
    aggressiveness: int,
    frame_ms: int,
    min_segment_ms: int,
    gap_merge_ms: int,
) -> List[Tuple[float, float]]:
    pcm, sr = _read_wave_bytes(wav_path)
    vad = webrtcvad.Vad(aggressiveness)
    voiced: List[Tuple[float, float]] = []
    current_start = None
    last_frame = None
    for ts, dur, b in _frame_generator(pcm, sr, frame_ms):
        is_voiced = vad.is_speech(b, sr)
        if is_voiced:
            if current_start is None:
                current_start = ts
        else:
            if current_start is not None:
                voiced.append((current_start, ts))
                current_start = None
        last_frame = (ts, dur)
    if current_start is not None and last_frame is not None:
        ts, dur = last_frame
        voiced.append((current_start, ts + dur))

    # merge and filter
    merged: List[Tuple[float, float]] = []
    for seg in voiced:
        if not merged:
            merged.append(seg)
            continue
        ps, pe = merged[-1]
        cs, ce = seg
        if (cs - pe) * 1000.0 <= gap_merge_ms:
            merged[-1] = (ps, ce)
        else:
            merged.append(seg)
    out: List[Tuple[float, float]] = []
    for s, e in merged:
        if (e - s) * 1000.0 >= min_segment_ms:
            out.append((s, e))
    return out


class VADStep(Step):
    name = "vad"

    def __init__(
        self,
        run_dir: str,
        aggressiveness: int = 2,
        min_ms: int = 150,
        merge_ms: int = 200,
    ):
        super().__init__(run_dir)
        self.aggr = aggressiveness
        self.min_ms = min_ms
        self.merge_ms = merge_ms

    def paths(self) -> Dict[str, str]:
        pdir = os.path.join(self.run_dir, "vad")
        ensure_dir(pdir)
        return {"segments": os.path.join(pdir, "segments.json")}

    def outputs_ready(self) -> bool:
        return file_info(self.paths()["segments"]).get("exists", False)

    def signature(self) -> Dict[str, Any]:
        wav = _detect_wav(self.run_dir)
        return {
            "wav": file_info(wav),
            "aggr": self.aggr,
            "min_ms": self.min_ms,
            "merge_ms": self.merge_ms,
        }

    def run(self) -> None:
        wav = _detect_wav(self.run_dir)
        logger.info(
            "[vad] wav=%s, aggr=%s, min_ms=%s, merge_ms=%s",
            wav,
            self.aggr,
            self.min_ms,
            self.merge_ms,
        )
        segs = _collect_vad_segments(
            wav,
            self.aggr,
            frame_ms=20,
            min_segment_ms=self.min_ms,
            gap_merge_ms=self.merge_ms,
        )
        logger.info("[vad] segments=%d", len(segs))
        write_json(self.paths()["segments"], {"segments": segs})


# ========== Score & events ==========


def _compute_scream_score(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    n_fft = 1024
    hop = 256
    y = y.astype(np.float32)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop, center=True)[0]
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, center=True
    )[0]
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, roll_percent=0.90, center=True
    )[0]
    flat = librosa.feature.spectral_flatness(
        y=y, n_fft=n_fft, hop_length=hop, center=True
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=n_fft, hop_length=hop, center=True
    )[0]

    def z(x):
        mu = float(np.mean(x))
        sd = float(np.std(x))
        sd = sd if sd > 1e-8 else 1e-8
        return (x - mu) / sd

    score = (
        0.30 * z(librosa.power_to_db(rms**2 + 1e-12))
        + 0.30 * z(centroid)
        + 0.25 * z(rolloff)
        + 0.10 * z(flat)
        + 0.05 * z(zcr)
    )
    if score.size >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5
        score = np.convolve(score, kernel, mode="same")
    times = librosa.frames_to_time(
        np.arange(score.shape[0]), sr=sr, hop_length=hop, n_fft=n_fft
    )
    return times, score


def _events_from_threshold(
    times: np.ndarray, score: np.ndarray, thr: float, min_dur: float, merge_gap: float
) -> List[Tuple[float, float, float]]:
    above = score >= thr
    events: List[Tuple[float, float, float]] = []
    i = 0
    n = len(times)
    while i < n:
        if not above[i]:
            i += 1
            continue
        s = times[i]
        mx = float(score[i])
        j = i + 1
        while j < n and above[j]:
            mx = max(mx, float(score[j]))
            j += 1
        e = times[min(j - 1, n - 1)]
        if e - s >= min_dur:
            events.append((s, e, mx))
        i = j
    # merge
    merged: List[Tuple[float, float, float]] = []
    for ev in events:
        if not merged:
            merged.append(ev)
            continue
        ps, pe, psc = merged[-1]
        cs, ce, csc = ev
        if cs - pe <= merge_gap:
            merged[-1] = (ps, ce, max(psc, csc))
        else:
            merged.append(ev)
    return merged


class ScoreStep(Step):
    name = "score"

    def __init__(
        self,
        run_dir: str,
        thr: float = 0.40,
        min_sec: float = 0.20,
        merge_gap: float = 0.30,
    ):
        super().__init__(run_dir)
        self.thr = thr
        self.min_sec = min_sec
        self.merge_gap = merge_gap

    def paths(self) -> Dict[str, str]:
        sdir = os.path.join(self.run_dir, "score")
        ensure_dir(sdir)
        return {
            "times": os.path.join(sdir, "times.npy"),
            "scores": os.path.join(sdir, "scores.npy"),
            "events": os.path.join(sdir, "events.json"),
        }

    def outputs_ready(self) -> bool:
        p = self.paths()
        return (
            file_info(p["times"]).get("exists", False)
            and file_info(p["scores"]).get("exists", False)
            and file_info(p["events"]).get("exists", False)
        )

    def signature(self) -> Dict[str, Any]:
        wav = _detect_wav(self.run_dir)
        return {
            "wav": file_info(wav),
            "thr": self.thr,
            "min_sec": self.min_sec,
            "merge_gap": self.merge_gap,
        }

    def run(self) -> None:
        wav = _detect_wav(self.run_dir)
        logger.info(
            "[score] wav=%s, thr=%.2f, min=%.2f, merge=%.2f",
            wav,
            self.thr,
            self.min_sec,
            self.merge_gap,
        )
        y, sr = librosa.load(wav, sr=16000, mono=True)
        times, scores = _compute_scream_score(y, sr)
        np.save(self.paths()["times"], times)
        np.save(self.paths()["scores"], scores)
        events = _events_from_threshold(
            times, scores, self.thr, self.min_sec, self.merge_gap
        )
        logger.info("[score] raw events=%d", len(events))
        write_json(self.paths()["events"], {"events": events})


class IntersectStep(Step):
    name = "intersect"

    def __init__(self, run_dir: str, pad_sec: float = 0.20):
        super().__init__(run_dir)
        self.pad = pad_sec

    def paths(self) -> Dict[str, str]:
        idir = os.path.join(self.run_dir, "events")
        ensure_dir(idir)
        return {"events_vad": os.path.join(idir, "events_vad.json")}

    def outputs_ready(self) -> bool:
        return file_info(self.paths()["events_vad"]).get("exists", False)

    def signature(self) -> Dict[str, Any]:
        vad = os.path.join(self.run_dir, "vad", "segments.json")
        ev = os.path.join(self.run_dir, "score", "events.json")
        return {"vad": file_info(vad), "events": file_info(ev), "pad": self.pad}

    def run(self) -> None:
        vad = read_json(os.path.join(self.run_dir, "vad", "segments.json")) or {
            "segments": []
        }
        events = read_json(os.path.join(self.run_dir, "score", "events.json")) or {
            "events": []
        }
        v = vad.get("segments", [])
        evs = events.get("events", [])
        logger.info(
            "[intersect] vad=%d, score_events=%d, pad=%.2f", len(v), len(evs), self.pad
        )
        # intersect with padding and merge overlaps
        out: List[Tuple[float, float, float]] = []
        for es, ee, scr in evs:
            for vs, ve in v:
                s = max(es, vs)
                e = min(ee, ve)
                if e - s > 0:
                    out.append((max(0.0, s - self.pad), e + self.pad, scr))
        out.sort(key=lambda x: x[0])
        merged: List[Tuple[float, float, float]] = []
        for ev in out:
            if not merged:
                merged.append(ev)
                continue
            ps, pe, pscr = merged[-1]
            cs, ce, cscr = ev
            if cs <= pe:
                merged[-1] = (ps, max(pe, ce), max(pscr, cscr))
            else:
                merged.append(ev)
        logger.info("[intersect] merged events=%d", len(merged))
        write_json(self.paths()["events_vad"], {"events": merged})


class PannsStep(Step):
    name = "panns"

    def __init__(
        self,
        run_dir: str,
        use_panns: bool = False,
        panns_thr: float = 0.25,
        anti_singing: bool = True,
    ):
        super().__init__(run_dir)
        self.use = use_panns
        self.thr = panns_thr
        self.anti = anti_singing

    def paths(self) -> Dict[str, str]:
        pdir = os.path.join(self.run_dir, "panns")
        ensure_dir(pdir)
        return {"refined": os.path.join(pdir, "refined.json")}

    def outputs_ready(self) -> bool:
        return file_info(self.paths()["refined"]).get("exists", False)

    def signature(self) -> Dict[str, Any]:
        ev = os.path.join(self.run_dir, "events", "events_vad.json")
        return {
            "events": file_info(ev),
            "use": self.use,
            "thr": self.thr,
            "anti": self.anti,
        }

    def run(self) -> None:
        ev = read_json(os.path.join(self.run_dir, "events", "events_vad.json")) or {
            "events": []
        }
        events = ev.get("events", [])
        if not self.use:
            logger.info("[panns] disabled; bypass (%d events)", len(events))
            write_json(self.paths()["refined"], {"events": events})
            return
        if not _HAS_PANNS:
            logger.warning("[panns] package not available; bypass")
            write_json(self.paths()["refined"], {"events": events})
            return
        if not events:
            logger.info("[panns] no events to refine")
            write_json(self.paths()["refined"], {"events": events})
            return
        # Use project-level shared models directory for checkpoints
        project_root = os.path.abspath(os.path.join(self.run_dir, os.pardir, os.pardir))
        ckpt_dir = os.path.join(project_root, "models", "panns")
        ensure_dir(ckpt_dir)
        ckpt_path = os.path.join(ckpt_dir, "Cnn14_mAP=0.431.pth")
        at = _PannsAT(checkpoint_path=ckpt_path, device="cpu")
        label_names = list(_PANN_LABELS)
        scream_idx = [
            i
            for i, n in enumerate(label_names)
            if any(k in n.lower() for k in ["scream", "yell", "shout"])
        ]
        music_idx = [
            i
            for i, n in enumerate(label_names)
            if any(
                k in n.lower()
                for k in ["music", "singing", "vocal music", "instrument"]
            )
        ]
        wav = _detect_wav(self.run_dir)
        logger.info("[panns] wav=%s, thr=%.2f, anti=%s", wav, self.thr, self.anti)
        y, sr = librosa.load(wav, sr=32000, mono=True)
        kept: List[Tuple[float, float, float]] = []
        for s, e, scr in events:
            s_s = int(max(0.0, s) * sr)
            e_s = int(max(0.0, e) * sr)
            seg = y[s_s:e_s].astype(np.float32)
            if seg.size < int(0.1 * sr):
                continue
            (clipwise_output, _emb) = at.inference(seg[None, :])
            probs = np.asarray(clipwise_output, dtype=np.float32)
            if probs.ndim == 2 and probs.shape[0] == 1:
                probs = probs[0]
            elif probs.ndim != 1:
                continue
            sc = float(np.max(probs[scream_idx])) if scream_idx else 0.0
            mc = float(np.max(probs[music_idx])) if music_idx else 0.0
            if sc >= self.thr and (not self.anti or mc < max(0.25, self.thr * 0.9)):
                kept.append((s, e, max(scr, sc)))
        logger.info("[panns] kept=%d / %d", len(kept), len(events))
        write_json(self.paths()["refined"], {"events": kept})


class ParalinguisticStep(Step):
    name = "paralinguistic"

    def __init__(
        self,
        run_dir: str,
        enable: bool = False,
        panns_enable: bool = False,
        panns_thr: float = 0.30,
    ):
        super().__init__(run_dir)
        self.enable = enable
        self.panns_enable = panns_enable
        self.panns_thr = panns_thr

    def paths(self) -> Dict[str, str]:
        pdir = os.path.join(self.run_dir, "paralinguistic")
        ensure_dir(pdir)
        return {"refined": os.path.join(pdir, "refined.json")}

    def outputs_ready(self) -> bool:
        return file_info(self.paths()["refined"]).get("exists", False)

    def signature(self) -> Dict[str, Any]:
        wav = _detect_wav(self.run_dir)
        vad = os.path.join(self.run_dir, "vad", "segments.json")
        return {
            "wav": file_info(wav),
            "vad": file_info(vad),
            "enable": self.enable,
            "panns_enable": self.panns_enable,
            "panns_thr": self.panns_thr,
        }

    def run(self) -> None:
        out_path = self.paths()["refined"]
        if not self.enable:
            logger.info("[quirks] disabled; writing empty categories")
            write_json(out_path, {"categories": {}})
            return
        wav = _detect_wav(self.run_dir)
        y, sr = librosa.load(wav, sr=16000, mono=True)
        # features
        n_fft = 1024
        hop = 256
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop, center=True)[
            0
        ]
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop, center=True
        )[0]
        flat = librosa.feature.spectral_flatness(
            y=y, n_fft=n_fft, hop_length=hop, center=True
        )[0]
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=n_fft, hop_length=hop, center=True
        )[0]
        rms_db = librosa.power_to_db(rms**2 + 1e-12)

        def z(x: np.ndarray) -> np.ndarray:
            mu = float(np.mean(x))
            sd = float(np.std(x))
            sd = sd if sd > 1e-8 else 1e-8
            return (x - mu) / sd

        zrms = z(rms_db)
        zcent = z(centroid)
        zflat = z(flat)
        zzcr = z(zcr)
        times = librosa.frames_to_time(
            np.arange(zrms.shape[0]), sr=sr, hop_length=hop, n_fft=n_fft
        )

        # VAD segments
        vad_json = read_json(os.path.join(self.run_dir, "vad", "segments.json")) or {
            "segments": []
        }
        vad_segments: List[Tuple[float, float]] = vad_json.get("segments", [])
        if not vad_segments:
            vad_segments = [(0.0, float(len(y)) / sr)]

        def collect_events(
            mask: np.ndarray, min_dur: float, merge_gap: float, score_arr: np.ndarray
        ) -> List[Tuple[float, float, float]]:
            events: List[Tuple[float, float, float]] = []
            i = 0
            n = len(mask)
            while i < n:
                if not mask[i]:
                    i += 1
                    continue
                s = times[i]
                mx = float(score_arr[i])
                j = i + 1
                while j < n and mask[j]:
                    mx = max(mx, float(score_arr[j]))
                    j += 1
                e = times[min(j - 1, n - 1)]
                if e - s >= min_dur:
                    events.append((s, e, mx))
                i = j
            # merge
            merged: List[Tuple[float, float, float]] = []
            for ev in events:
                if not merged:
                    merged.append(ev)
                    continue
                ps, pe, psc = merged[-1]
                cs, ce, csc = ev
                if cs - pe <= merge_gap:
                    merged[-1] = (ps, ce, max(psc, csc))
                else:
                    merged.append(ev)
            return merged

        categories: Dict[str, List[Tuple[float, float, float]]] = {}

        # Click/Lip smack: short, high flatness + high centroid spikes
        click_mask = (zflat > 1.0) & (zcent > 0.5) & (zrms > -1.0)
        click_events_all = collect_events(
            click_mask,
            min_dur=0.02,
            merge_gap=0.05,
            score_arr=(0.6 * zflat + 0.4 * zcent),
        )
        # Restrict to VAD windows
        clicks: List[Tuple[float, float, float]] = []
        for s, e, sc in click_events_all:
            for vs, ve in vad_segments:
                ss, ee = max(s, vs), min(e, ve)
                if ee - ss > 0:
                    clicks.append((ss, ee, sc))
        categories["click"] = clicks

        # Murmur: low energy voiced, lower flatness, moderate zcr, short to mid duration
        mur_mask = (zrms > -1.0) & (zrms < 0.5) & (zflat < 0.0) & (zzcr < 0.5)
        murm_all = collect_events(
            mur_mask,
            min_dur=0.10,
            merge_gap=0.20,
            score_arr=(0.5 * (-zflat) + 0.5 * (-zzcr)),
        )
        murm: List[Tuple[float, float, float]] = []
        for s, e, sc in murm_all:
            for vs, ve in vad_segments:
                ss, ee = max(s, vs), min(e, ve)
                if ee - ss > 0:
                    murm.append((ss, ee, sc))
        categories["murmur"] = murm

        # Sigh: longer, noisy-ish (flatness high), centroid lower, duration >= 0.3s
        sigh_mask = (zflat > 0.5) & (zcent < 0.0)
        sigh_all = collect_events(
            sigh_mask,
            min_dur=0.30,
            merge_gap=0.30,
            score_arr=(0.7 * zflat + 0.3 * (-zcent)),
        )
        sighs: List[Tuple[float, float, float]] = []
        for s, e, sc in sigh_all:
            for vs, ve in vad_segments:
                ss, ee = max(s, vs), min(e, ve)
                if ee - ss > 0:
                    sighs.append((ss, ee, sc))
        categories["sigh"] = sighs

        # Optional: PANNs-based categories from AudioSet labels
        if self.panns_enable and _HAS_PANNS:
            # Prepare model and labels
            project_root = os.path.abspath(
                os.path.join(self.run_dir, os.pardir, os.pardir)
            )
            ckpt_dir = os.path.join(project_root, "models", "panns")
            ensure_dir(ckpt_dir)
            ckpt_path = os.path.join(ckpt_dir, "Cnn14_mAP=0.431.pth")
            at = _PannsAT(checkpoint_path=ckpt_path, device="cpu")
            label_names = list(_PANN_LABELS)

            def idx_for(keys: List[str]) -> List[int]:
                out_idx: List[int] = []
                for i, n in enumerate(label_names):
                    ln = n.lower()
                    if any(k in ln for k in keys):
                        out_idx.append(i)
                return out_idx

            label_map = {
                "hum": idx_for(["hum", "humming"]),
                "whisper": idx_for(["whisper"]),
                "breathing": idx_for(["breathing", "breath"]),
                "throat_clearing": idx_for(["throat clearing"]),
                "cough": idx_for(["cough"]),
                "gasp": idx_for(["gasp"]),
                "groan": idx_for(["groan"]),
                "sigh": idx_for(["sigh"]),
            }

            # Slide over VAD segments with chunking to improve time localization
            y32, sr32 = librosa.load(wav, sr=32000, mono=True)
            chunk_len = 1.5
            hop = 0.75
            min_len_sec = 1.5  # pad shorter chunks to this length for CNN14 stability
            for vs, ve in vad_segments:
                t = vs
                while t < ve:
                    ce = min(ve, t + chunk_len)
                    if ce - t < 0.3:
                        break
                    s_s = int(t * sr32)
                    e_s = int(ce * sr32)
                    seg = y32[s_s:e_s].astype(np.float32)
                    # Pad too-short segments to minimum length
                    min_samples = int(min_len_sec * sr32)
                    if seg.size < min_samples:
                        pad = np.zeros(max(0, min_samples - seg.size), dtype=np.float32)
                        seg = np.concatenate([seg, pad], axis=0)
                    (clipwise_output, _emb) = at.inference(seg[None, :])
                    probs = np.asarray(clipwise_output, dtype=np.float32)
                    if probs.ndim == 2 and probs.shape[0] == 1:
                        probs = probs[0]
                    elif probs.ndim != 1:
                        t += hop
                        continue
                    for cat, indices in label_map.items():
                        if not indices:
                            continue
                        conf = float(np.max(probs[indices]))
                        if conf >= self.panns_thr:
                            categories.setdefault(cat, []).append((t, ce, conf))
                    t += hop

        # Merge overlaps per category for cleanliness
        def merge_cat(
            evs: List[Tuple[float, float, float]], gap: float
        ) -> List[Tuple[float, float, float]]:
            if not evs:
                return evs
            evs = sorted(evs, key=lambda x: x[0])
            out: List[Tuple[float, float, float]] = [evs[0]]
            for s, e, sc in evs[1:]:
                ps, pe, psc = out[-1]
                if s - pe <= gap:
                    out[-1] = (ps, max(pe, e), max(psc, sc))
                else:
                    out.append((s, e, sc))
            return out

        for k in list(categories.keys()):
            categories[k] = merge_cat(categories[k], gap=0.20)

        logger.info(
            "[quirks] click=%d, murmur=%d, sigh=%d, +panns=%s",
            len(clicks),
            len(murm),
            len(sighs),
            "on" if (self.panns_enable and _HAS_PANNS) else "off",
        )
        write_json(out_path, {"categories": categories})


class ExportStep(Step):
    name = "export"

    def __init__(self, run_dir: str, video_path: str, reencode: bool = False):
        super().__init__(run_dir)
        self.video = video_path
        self.reencode = reencode

    def paths(self) -> Dict[str, str]:
        edir = os.path.join(self.run_dir, "export")
        ensure_dir(edir)
        ensure_dir(os.path.join(edir, "clips"))
        return {
            "csv_all": os.path.join(edir, "events.csv"),
        }

    def outputs_ready(self) -> bool:
        p = self.paths()
        return file_info(p["csv_all"]).get("exists", False)

    def signature(self) -> Dict[str, Any]:
        # Prefer PANNs refined if exists
        panns_ev = os.path.join(self.run_dir, "panns", "refined.json")
        quirks_ev = os.path.join(self.run_dir, "paralinguistic", "refined.json")
        base_ev = os.path.join(self.run_dir, "events", "events_vad.json")
        return {
            "events_sources": [
                file_info(panns_ev),
                file_info(quirks_ev),
                file_info(base_ev),
            ],
            "video": file_info(self.video),
            "reencode": self.reencode,
        }

    def run(self) -> None:
        panns_ev = os.path.join(self.run_dir, "panns", "refined.json")
        base_ev = os.path.join(self.run_dir, "events", "events_vad.json")
        quirks_ev = os.path.join(self.run_dir, "paralinguistic", "refined.json")

        # Aggregate events with type
        events_all: List[Tuple[float, float, float, str]] = []
        if os.path.isfile(panns_ev):
            data = read_json(panns_ev) or {"events": []}
            for s, e, sc in data.get("events", []):
                events_all.append((s, e, sc, "scream"))
        else:
            data = read_json(base_ev) or {"events": []}
            for s, e, sc in data.get("events", []):
                events_all.append((s, e, sc, "scream"))

        if os.path.isfile(quirks_ev):
            qdata = read_json(quirks_ev) or {"categories": {}}
            cats = qdata.get("categories", {})
            for cat, evs in cats.items():
                for s, e, sc in evs:
                    events_all.append((s, e, sc, cat))

        # TEMPORARY: enforce minimal export duration to reduce noise clips
        # This is a temporary safeguard requested by user to ensure that
        # exported events are at least 0.5 seconds long at the export layer.
        MIN_EXPORT_DUR = 0.5
        events_all = [
            (s, e, sc, typ)
            for (s, e, sc, typ) in events_all
            if (e - s) >= MIN_EXPORT_DUR
        ]

        # Write combined CSV (with Chinese type and mm:ss text time)
        p = self.paths()
        import csv

        def fmt_time(t: float) -> str:
            if t < 0:
                t = 0.0
            total_sec = int(round(t))
            mm = total_sec // 60
            ss = total_sec % 60
            return f"{mm:02d}:{ss:02d}"

        type_zh = {
            "scream": "尖叫",
            "click": "咂嘴/口腔点击",
            "murmur": "嘟囔",
            "sigh": "叹气",
            "hum": "哼声",
            "whisper": "耳语/低语",
            "breathing": "呼吸",
            "throat_clearing": "清嗓",
            "cough": "咳嗽",
            "gasp": "倒吸气/喘息",
            "groan": "呻吟",
        }
        with open(p["csv_all"], "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                ["type", "type_zh", "start", "end", "start_text", "end_text", "score"]
            )
            for s, e, sc, typ in events_all:
                w.writerow(
                    [
                        typ,
                        type_zh.get(typ, typ),
                        f"{s:.3f}",
                        f"{e:.3f}",
                        fmt_time(s),
                        fmt_time(e),
                        f"{sc:.4f}",
                    ]
                )

        # Per-category cut scripts and categorized clips
        edir = os.path.join(self.run_dir, "export")
        by_cat: Dict[str, List[Tuple[float, float]]] = {}
        for s, e, _sc, typ in events_all:
            by_cat.setdefault(typ, []).append((s, e))

        for typ, evs in by_cat.items():
            clips_dir = os.path.join(edir, "clips", typ)
            ensure_dir(clips_dir)
            sh_path = os.path.join(edir, f"cut_{typ}.sh")
            lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
            for s, e in evs:
                dur = max(0.0, e - s)
                out = os.path.join(clips_dir, f"{fmt_time(s)}.mp4")
                if self.reencode:
                    cmd = f"ffmpeg -y -ss {s:.3f} -i '{self.video}' -t {dur:.3f} -c:v libx264 -crf 18 -preset veryfast -c:a aac -b:a 160k '{out}'"
                else:
                    cmd = f"ffmpeg -y -ss {s:.3f} -i '{self.video}' -t {dur:.3f} -c copy '{out}'"
                lines.append(cmd)
            with open(sh_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            os.chmod(sh_path, 0o755)
            logger.info("[export] %s: %d clips, sh=%s", typ, len(evs), sh_path)
