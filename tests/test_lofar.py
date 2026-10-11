"""LOFAR tests (F2-T05) with synthetic signals whose answer is known in advance."""

import itertools

import numpy as np
import pytest
from scipy.signal import lfilter

from sonar_pipeline.lofar import lofar, lofar_frequencies, tpsw
from sonar_pipeline.params import SAMPLE_RATE
from sonar_pipeline.preprocess import preprocess

FS = SAMPLE_RATE
N = 3 * FS  # one 3 s segment
T = np.arange(N) / FS
FREQS = lofar_frequencies()
BIN_HZ = FREQS[1] - FREQS[0]


def noise(seed=0):
    return np.random.default_rng(seed).standard_normal(N)


def segment_of(x):
    """Go through the real preprocessing, as the pipeline does."""
    return preprocess(x)[0]


def band_means(gram, n_bands=10):
    """Mean linear power ratio in equal-width frequency bands."""
    ratio = 10 ** (gram / 10)
    edges = np.linspace(0, FREQS[-1], n_bands + 1)
    return np.array(
        [
            ratio[:, (FREQS >= lo) & (FREQS < hi)].mean()
            for lo, hi in itertools.pairwise(edges)
        ]
    )


def test_shape_and_frequency_axis():
    # D-015: 11 frames x 1001 bins of 2 Hz from 0 to 2 kHz.
    assert BIN_HZ == pytest.approx(2.0)
    assert FREQS[0] == 0.0 and FREQS[-1] == pytest.approx(2000.0)
    gram = lofar(segment_of(noise()))
    assert gram.shape == (11, 1001)
    assert np.all(np.isfinite(gram))


@pytest.mark.parametrize(
    ("f0", "tolerance_hz"),
    # On a bin centre the peak must be exactly there; between two bins, either neighbour.
    [(440.0, 0.0), (1950.0, 0.0), (333.0, BIN_HZ), (1234.5, BIN_HZ)],
)
def test_known_tone_peaks_at_its_frequency_in_every_frame(f0, tolerance_hz):
    # A weak tone (amplitude 0.1) buried in unit white noise, also near the 2 kHz band edge.
    gram = lofar(segment_of(0.1 * np.sin(2 * np.pi * f0 * T) + noise()))
    for frame in gram:
        assert abs(FREQS[np.argmax(frame)] - f0) <= tolerance_hz
    assert gram.mean(axis=0).max() > 10.0  # clearly above the background (dB)


@pytest.mark.parametrize("f0", [2050.0, 2100.0, 2200.0, 2600.0, 3000.0])
def test_tone_above_band_does_not_alias(f0):
    # Above the 2 kHz kept, a strong tone must leave no line at all: not folded near the band
    # edge (2050 and 2100 Hz did with decimation to 4 kHz, F2 review M1) nor further down.
    gram = lofar(segment_of(0.3 * np.sin(2 * np.pi * f0 * T) + noise()))
    assert gram.mean(axis=0).max() < 6.0  # an in-band tone of this level gives > 20 dB


@pytest.mark.parametrize("colour", ["white", "red"])
def test_normalised_noise_is_flat(colour):
    # TPSW removes the background shape: white noise and strongly coloured (red) noise both
    # give the same level in every band. That level is a constant ~1.4 (about +1.4 dB) and
    # not 1: the second TPSW pass clips noise peaks above 2x the mean, which lowers the
    # background estimate. A constant offset does not change the shape.
    x = noise() if colour == "white" else lfilter([1.0], [1.0, -0.95], noise())
    gram = lofar(segment_of(x))
    means = band_means(gram)
    assert np.all(np.abs(means / means.mean() - 1) < 0.1)
    assert 1.2 < means.mean() < 1.6
    assert abs(gram.mean()) < 2.0  # dB, about 0


def test_tpsw_ignores_a_tonal_line():
    power = np.ones((1, 201))
    power[0, 100] = 1000.0
    background = tpsw(power)
    np.testing.assert_allclose(background, 1.0)


def test_tpsw_follows_a_sloped_background():
    power = np.linspace(1.0, 3.0, 201)[None, :]
    np.testing.assert_allclose(tpsw(power)[0, 30:-30], power[0, 30:-30], rtol=1e-9)
