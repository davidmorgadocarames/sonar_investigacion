"""DEMON (D-015): spectrum of the envelope of the cavitation band of one segment.

Propeller cavitation is broadband noise whose amplitude is modulated at the shaft and blade
rates. Steps: zero-phase Butterworth band-pass (2-8 kHz), |Hilbert| envelope minus its mean,
decimation to 500 Hz in two x8 stages (polyphase FIR anti-aliasing), Hann magnitude spectrum of
the whole segment (1/3 Hz bins from 0 to 100 Hz), bins below 1 Hz set to 0, scaled to max 1.
"""

import numpy as np
from scipy.signal import butter, get_window, hilbert, resample_poly, sosfiltfilt

from sonar_pipeline.params import (
    DEMON_BAND_HZ,
    DEMON_DECIMATION,
    DEMON_FILTER_ORDER,
    DEMON_MAX_HZ,
    DEMON_MIN_HZ,
    SAMPLE_RATE,
)


def envelope_rate(sample_rate: int = SAMPLE_RATE) -> float:
    return sample_rate / np.prod(DEMON_DECIMATION)


def demon_frequencies(n_samples: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Frequency (Hz) of every DEMON bin kept, for a segment of n_samples at sample_rate."""
    n_envelope = n_samples
    for factor in DEMON_DECIMATION:
        # resample_poly output length: ceil(n / q)
        n_envelope = -(-n_envelope // factor)
    freqs = np.fft.rfftfreq(n_envelope, d=1 / envelope_rate(sample_rate))
    return freqs[freqs <= DEMON_MAX_HZ + 1e-9]


def envelope(segment: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Band-pass, |Hilbert| envelope minus its mean, decimated to the envelope rate."""
    sos = butter(
        DEMON_FILTER_ORDER,
        DEMON_BAND_HZ,
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    env = np.abs(hilbert(sosfiltfilt(sos, segment)))
    env = env - np.mean(env)
    for factor in DEMON_DECIMATION:
        env = resample_poly(env, up=1, down=factor)
    return env


def demon(segment: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """One preprocessed segment -> DEMON spectrum scaled to max 1, shape (n_bins,)."""
    env = envelope(segment, sample_rate)
    spectrum = np.abs(np.fft.rfft(env * get_window("hann", len(env))))
    freqs = demon_frequencies(len(segment), sample_rate)
    expected = np.fft.rfftfreq(len(env), d=1 / envelope_rate(sample_rate))
    if not np.array_equal(freqs, expected[: len(freqs)]):
        raise ValueError("DEMON frequency axis does not match the envelope length")
    spectrum = spectrum[: len(freqs)]
    spectrum[freqs < DEMON_MIN_HZ] = 0.0
    peak = spectrum.max()
    if peak == 0:
        raise ValueError(
            "DEMON spectrum is zero: the envelope has no modulation at all"
        )
    return spectrum / peak
