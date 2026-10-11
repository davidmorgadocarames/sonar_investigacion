"""Preprocessing (D-009, D-015): remove the DC offset per recording, cut fixed-length segments
without overlap and scale each segment to unit RMS.

Segments are always cut from a single recording: the functions take one recording at a time.
"""

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from sonar_pipeline.params import SAMPLE_RATE, SEGMENT_S
from sonar_pipeline.qc import to_full_scale


def load_wav(path: Path) -> np.ndarray:
    """Read a mono WAV at the expected sample rate as float64 in full-scale units."""
    sample_rate, samples = wavfile.read(path)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path}: sample rate {sample_rate}, expected {SAMPLE_RATE}")
    if samples.ndim != 1:
        raise ValueError(f"{path}: {samples.shape[1]} channels, expected 1")
    x = to_full_scale(samples)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{path}: non-finite samples")
    return x


def remove_dc(x: np.ndarray) -> np.ndarray:
    """Subtract the mean of the whole recording (not of each segment)."""
    return x - np.mean(x)


def segment_length(sample_rate: int = SAMPLE_RATE, seconds: float = SEGMENT_S) -> int:
    n = seconds * sample_rate
    if n != int(n) or n <= 0:
        raise ValueError(
            f"{seconds} s at {sample_rate} Hz is not a whole number of samples"
        )
    return int(n)


def segment(
    x: np.ndarray, sample_rate: int = SAMPLE_RATE, seconds: float = SEGMENT_S
) -> np.ndarray:
    """Cut consecutive segments without overlap; the incomplete tail is dropped.

    Returns an array of shape (n_segments, segment_length). Segment i starts at sample
    i * segment_length. Fails if the recording does not fill one segment (D-009).
    """
    length = segment_length(sample_rate, seconds)
    n = len(x) // length
    if n == 0:
        raise ValueError(
            f"recording of {len(x) / sample_rate:.3f} s is shorter than one {seconds} s segment"
        )
    return x[: n * length].reshape(n, length)


def normalize_rms(segments: np.ndarray) -> np.ndarray:
    """Scale each segment (row) to RMS = 1. A silent segment cannot be scaled and fails."""
    rms = np.sqrt(np.mean(segments**2, axis=1, keepdims=True))
    if np.any(rms == 0):
        raise ValueError("silent segment (RMS = 0) cannot be normalised")
    return segments / rms


def preprocess(
    x: np.ndarray, sample_rate: int = SAMPLE_RATE, seconds: float = SEGMENT_S
) -> np.ndarray:
    """One recording -> normalised segments of shape (n_segments, segment_length)."""
    return normalize_rms(segment(remove_dc(x), sample_rate, seconds))
