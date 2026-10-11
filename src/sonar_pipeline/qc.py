"""Quality control of raw WAV files.

QC only measures and flags. It never excludes or modifies a file: what to do with flagged
recordings is a separate, approved decision (task F1-T05b).
"""

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# DeepShip subset format, measured in the F1-T03 inventory.
EXPECTED_SAMPLE_RATE = 32000
EXPECTED_CHANNELS = 1
EXPECTED_DTYPE = "float32"

# A float WAV is full scale at |x| = 1.0; integer WAVs at their dtype limits.
CLIP_LEVEL_FLOAT = 1.0
# Flag DC offset when |mean| exceeds this fraction of the AC standard deviation.
DC_RATIO_THRESHOLD = 0.1
# Flag silence when the standard deviation is below this (in full-scale units).
SILENCE_STD_THRESHOLD = 1e-6


@dataclass
class QCResult:
    path: str
    vessel_class: str
    sample_rate: int
    channels: int
    dtype: str
    n_samples: int
    duration_s: float
    dc_offset: float
    std: float
    dc_ratio: float
    peak: float
    n_clipped: int
    n_nonfinite: int
    flag_sample_rate: bool
    flag_channels: bool
    flag_dtype: bool
    flag_dc: bool
    flag_clipping: bool
    flag_nonfinite: bool
    flag_silent: bool

    @property
    def flagged(self) -> bool:
        return any(v for k, v in asdict(self).items() if k.startswith("flag_"))


def to_full_scale(samples: np.ndarray) -> np.ndarray:
    """Convert samples to float64 where full scale is 1.0."""
    if np.issubdtype(samples.dtype, np.floating):
        return samples.astype(np.float64)
    if samples.dtype == np.uint8:
        return (samples.astype(np.float64) - 128.0) / 128.0
    return samples.astype(np.float64) / float(-np.iinfo(samples.dtype).min)


def analyze(samples: np.ndarray, sample_rate: int) -> dict:
    """Measure one recording. Multichannel input is analysed on all channels together."""
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    n_samples = samples.shape[0]
    x = to_full_scale(samples).ravel()

    finite = np.isfinite(x)
    n_nonfinite = int(np.size(x) - np.count_nonzero(finite))
    x = x[finite]

    if x.size:
        dc_offset = float(np.mean(x))
        std = float(np.std(x))
        peak = float(np.max(np.abs(x)))
    else:
        dc_offset = std = peak = 0.0
    dc_ratio = abs(dc_offset) / std if std > 0 else (float("inf") if dc_offset else 0.0)

    if np.issubdtype(samples.dtype, np.floating):
        n_clipped = int(np.count_nonzero(np.abs(x) >= CLIP_LEVEL_FLOAT))
    else:
        raw = samples.ravel()
        info = np.iinfo(samples.dtype)
        n_clipped = int(np.count_nonzero((raw == info.min) | (raw == info.max)))

    return {
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "dtype": str(samples.dtype),
        "n_samples": int(n_samples),
        "duration_s": n_samples / sample_rate if sample_rate else 0.0,
        "dc_offset": dc_offset,
        "std": std,
        "dc_ratio": dc_ratio,
        "peak": peak,
        "n_clipped": n_clipped,
        "n_nonfinite": n_nonfinite,
        "flag_sample_rate": sample_rate != EXPECTED_SAMPLE_RATE,
        "flag_channels": channels != EXPECTED_CHANNELS,
        "flag_dtype": str(samples.dtype) != EXPECTED_DTYPE,
        "flag_dc": dc_ratio > DC_RATIO_THRESHOLD,
        "flag_clipping": n_clipped > 0,
        "flag_nonfinite": n_nonfinite > 0,
        "flag_silent": std < SILENCE_STD_THRESHOLD,
    }


def check_file(path: Path, data_dir: Path) -> QCResult:
    sample_rate, samples = wavfile.read(path)
    rel = path.relative_to(data_dir)
    vessel_class = rel.parts[0] if len(rel.parts) > 1 else ""
    return QCResult(
        path=rel.as_posix(), vessel_class=vessel_class, **analyze(samples, sample_rate)
    )


def natural_key(path: str) -> list:
    """Sort key that orders numbers by value: Cargo/9.wav before Cargo/10.wav."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path)]


def run_qc(data_dir: Path, output: Path) -> list[QCResult]:
    """Check every WAV under data_dir (natural order, recursive); write one LF CSV row per file."""
    wavs = sorted(
        (
            p
            for p in data_dir.rglob("*")
            if p.suffix.lower() == ".wav" and ".git" not in p.parts
        ),
        key=lambda p: natural_key(p.relative_to(data_dir).as_posix()),
    )
    if not wavs:
        raise FileNotFoundError(f"No WAV files found under {data_dir}")
    results = [check_file(p, data_dir) for p in wavs]

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [*QCResult.__dataclass_fields__, "flagged"]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for r in results:
            writer.writerow({**asdict(r), "flagged": r.flagged})
    return results
