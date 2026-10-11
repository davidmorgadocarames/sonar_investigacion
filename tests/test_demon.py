"""DEMON tests (F2-T06) with synthetic signals whose answer is known in advance."""

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from sonar_pipeline.demon import demon, demon_frequencies
from sonar_pipeline.params import SAMPLE_RATE
from sonar_pipeline.preprocess import preprocess

FS = SAMPLE_RATE
N = 3 * FS  # one 3 s segment
T = np.arange(N) / FS
FREQS = demon_frequencies(N)
BIN_HZ = FREQS[1] - FREQS[0]
SEEDS = [0, 1, 2]


def noise(seed):
    return np.random.default_rng(seed).standard_normal(N)


def modulated(fm, depth, seed):
    """Broadband noise with its amplitude modulated at fm, like propeller cavitation."""
    return (1 + depth * np.cos(2 * np.pi * fm * T)) * noise(seed)


def spectrum_of(x):
    """Go through the real preprocessing, as the pipeline does."""
    return demon(preprocess(x)[0])


def prominence(spectrum):
    """Highest bin divided by the median bin, above the 1 Hz cut."""
    kept = spectrum[FREQS >= 1.0]
    return kept.max() / np.median(kept)


def test_shape_and_frequency_axis():
    # D-015: 1/3 Hz bins from 0 to 100 Hz, bins below 1 Hz set to 0, max 1.
    assert len(FREQS) == 301
    assert BIN_HZ == pytest.approx(1 / 3)
    assert FREQS[-1] == pytest.approx(100.0)
    spectrum = spectrum_of(noise(0))
    assert spectrum.shape == (301,)
    assert np.all(spectrum[FREQS < 1.0] == 0.0)
    assert spectrum.max() == pytest.approx(1.0)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("fm", [3.4, 7.5, 12.0, 50.0])
def test_known_modulation_peaks_at_its_frequency(fm, seed):
    # Shaft-like (3.4 Hz) to blade-like (50 Hz) rates, also between two bins (7.5 Hz).
    spectrum = spectrum_of(modulated(fm, depth=0.5, seed=seed))
    assert abs(FREQS[np.argmax(spectrum)] - fm) <= BIN_HZ
    assert prominence(spectrum) > 20


@pytest.mark.parametrize("seed", SEEDS)
def test_weak_modulation_is_still_found(seed):
    spectrum = spectrum_of(modulated(12.0, depth=0.1, seed=seed))
    assert abs(FREQS[np.argmax(spectrum)] - 12.0) <= BIN_HZ
    assert prominence(spectrum) > 8


@pytest.mark.parametrize("seed", SEEDS)
def test_unmodulated_noise_has_no_peak(seed):
    assert prominence(spectrum_of(noise(seed))) < 5


@pytest.mark.parametrize("seed", SEEDS)
def test_modulation_outside_the_band_is_ignored(seed):
    # Strong 12 Hz modulation of noise below 1 kHz, plus unmodulated noise everywhere: the
    # 2-8 kHz band-pass removes the modulated part, so no peak appears.
    rng = np.random.default_rng(seed + 100)
    low = sosfiltfilt(butter(4, 1000, fs=FS, output="sos"), rng.standard_normal(N))
    x = 5 * (1 + 0.9 * np.cos(2 * np.pi * 12.0 * T)) * low + noise(seed)
    spectrum = spectrum_of(x)
    assert prominence(spectrum) < 5
    assert abs(FREQS[np.argmax(spectrum)] - 12.0) > BIN_HZ


@pytest.mark.parametrize("fm", [400.0, 450.0])
def test_modulation_above_100_hz_does_not_alias(fm):
    # Decimating the envelope to 500 Hz would fold 400 and 450 Hz onto 100 and 50 Hz without
    # anti-aliasing: no peak may appear in 0-100 Hz.
    assert prominence(spectrum_of(modulated(fm, depth=0.9, seed=0))) < 5


@pytest.mark.parametrize("fm", [12.0, 30.0])
def test_modulation_on_a_bin_centre_peaks_exactly_there(fm):
    spectrum = spectrum_of(modulated(fm, depth=0.5, seed=0))
    assert FREQS[np.argmax(spectrum)] == pytest.approx(fm)


def test_constant_envelope_fails():
    with pytest.raises(ValueError, match="no modulation"):
        demon(np.zeros(N))
