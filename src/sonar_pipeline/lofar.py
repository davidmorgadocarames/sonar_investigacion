"""LOFAR (D-015): narrowband spectrogram of one segment, normalised by its noise background.

Steps: decimate to 4 kHz (polyphase FIR anti-aliasing), Hann STFT of 0.5 s with a 0.25 s hop
(2 Hz bins from 0 to 2 kHz), power, TPSW background estimate along frequency in every frame,
and 10*log10(power / background): dB above the background, about 0 dB for flat noise.
"""

import numpy as np
from scipy.signal import get_window, resample_poly

from sonar_pipeline.params import (
    LOFAR_HOP_S,
    LOFAR_MAX_HZ,
    LOFAR_SAMPLE_RATE,
    LOFAR_WINDOW_S,
    SAMPLE_RATE,
    TPSW_CLIP_FACTOR,
    TPSW_GAP_BINS,
    TPSW_HALF_WINDOW_BINS,
)

# Numerical floor for power ratios, so that an empty bin gives -100 dB instead of -inf.
MIN_RATIO = 1e-10


def window_length() -> int:
    return round(LOFAR_WINDOW_S * LOFAR_SAMPLE_RATE)


def hop_length() -> int:
    return round(LOFAR_HOP_S * LOFAR_SAMPLE_RATE)


def lofar_frequencies() -> np.ndarray:
    """Centre frequency (Hz) of every LOFAR bin kept, from 0 to LOFAR_MAX_HZ."""
    freqs = np.fft.rfftfreq(window_length(), d=1 / LOFAR_SAMPLE_RATE)
    return freqs[freqs <= LOFAR_MAX_HZ]


def decimate(x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    if sample_rate % LOFAR_SAMPLE_RATE:
        raise ValueError(
            f"{sample_rate} Hz is not a multiple of {LOFAR_SAMPLE_RATE} Hz"
        )
    return resample_poly(x, up=1, down=sample_rate // LOFAR_SAMPLE_RATE)


def power_spectrogram(x: np.ndarray) -> np.ndarray:
    """Hann STFT power of a 4 kHz signal: shape (n_frames, n_bins), frames without padding."""
    n, hop = window_length(), hop_length()
    if len(x) < n:
        raise ValueError(
            f"signal of {len(x)} samples is shorter than one {n}-sample window"
        )
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    spectrum = np.fft.rfft(frames * get_window("hann", n), axis=1)
    return np.abs(spectrum[:, : len(lofar_frequencies())]) ** 2


def split_window_mean(x: np.ndarray, half_window: int, gap: int) -> np.ndarray:
    """Mean of the neighbours of every bin along the last axis, skipping a central gap.

    Near the edges only the neighbours that exist are averaged.
    """
    kernel = np.ones(2 * half_window + 1)
    kernel[half_window - gap : half_window + gap + 1] = 0.0
    ones = np.ones(x.shape[-1])
    count = np.convolve(ones, kernel, mode="same")

    def smooth(row):
        return np.convolve(row, kernel, mode="same") / count

    return np.apply_along_axis(smooth, -1, x)


def tpsw(
    power: np.ndarray,
    half_window: int = TPSW_HALF_WINDOW_BINS,
    gap: int = TPSW_GAP_BINS,
    clip_factor: float = TPSW_CLIP_FACTOR,
) -> np.ndarray:
    """Two-Pass Split Window background estimate along frequency (last axis).

    First pass: split-window mean. Values above clip_factor times that mean (tonal lines) are
    replaced by the mean, and a second split-window mean gives the background.
    """
    first = split_window_mean(power, half_window, gap)
    clipped = np.where(power > clip_factor * first, first, power)
    return split_window_mean(clipped, half_window, gap)


def lofar(segment: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """One preprocessed segment -> LOFARgram in dB above the background, (n_frames, n_bins)."""
    power = power_spectrogram(decimate(segment, sample_rate))
    background = tpsw(power)
    ratio = power / np.maximum(background, np.finfo(np.float64).tiny)
    return 10.0 * np.log10(np.maximum(ratio, MIN_RATIO))
