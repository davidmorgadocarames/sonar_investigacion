"""Preprocessing tests (F2-T04) with synthetic signals, plus a check on the real WAVs if present."""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from sonar_pipeline.params import SAMPLE_RATE, SEGMENT_S
from sonar_pipeline.preprocess import (
    load_wav,
    normalize_rms,
    preprocess,
    remove_dc,
    segment,
    segment_length,
)

FS = SAMPLE_RATE
L = int(SEGMENT_S * FS)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "raw" / "DeepShip"
SPLIT = ROOT / "datasets" / "manifests" / "split.csv"


def tone(seconds, freq=100.0, amplitude=0.01, offset=0.0):
    t = np.arange(round(seconds * FS)) / FS
    return offset + amplitude * np.sin(2 * np.pi * freq * t)


def test_segment_length_is_three_seconds():
    assert segment_length() == 96000


def test_dc_offset_disappears():
    # Like DeepShip: an offset of 0.03 larger than the signal itself.
    x = tone(10.0, offset=0.03)
    y = remove_dc(x)
    assert np.mean(y) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(y, x - 0.03, atol=1e-12)
    for s in preprocess(x):
        assert abs(np.mean(s)) < 1e-3


def test_dc_is_removed_per_recording_not_per_segment():
    # Recording mean is 0, but each half has its own mean: they must be kept.
    x = np.concatenate([np.full(L, 1.0), np.full(L, -1.0)])
    np.testing.assert_array_equal(remove_dc(x), x)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(3.0, 1), (5.999, 1), (6.0, 2), (10.0, 3), (204.0, 68)],
)
def test_number_of_segments_is_exact(seconds, expected):
    assert segment(tone(seconds)).shape == (expected, L)


def test_segments_are_consecutive_slices_of_the_recording():
    x = np.arange(int(10.5 * FS), dtype=np.float64)
    segs = segment(x)
    for i, s in enumerate(segs):
        np.testing.assert_array_equal(s, x[i * L : (i + 1) * L])
    # The 1.5 s tail is dropped, nothing else.
    assert segs.size == 3 * L
    assert segs[-1][-1] == 3 * L - 1


def test_no_segment_crosses_recordings():
    # Each recording is segmented on its own: tails are dropped, never joined.
    a = np.full(int(4.0 * FS), 1.0)
    b = np.full(int(4.0 * FS), 2.0)
    segs_a, segs_b = segment(a), segment(b)
    assert len(segs_a) == len(segs_b) == 1
    assert np.all(segs_a == 1.0) and np.all(segs_b == 2.0)


def test_recording_shorter_than_one_segment_fails():
    with pytest.raises(ValueError, match="shorter than one"):
        segment(tone(2.999))


def test_each_segment_has_unit_rms():
    # Amplitude grows between segments; after normalisation every segment has RMS 1.
    x = np.concatenate([tone(3.0, amplitude=a) for a in (0.001, 0.01, 0.5)])
    segs = preprocess(x)
    np.testing.assert_allclose(np.sqrt(np.mean(segs**2, axis=1)), 1.0, rtol=1e-12)


def test_silent_segment_fails():
    with pytest.raises(ValueError, match="silent"):
        normalize_rms(np.zeros((1, L)))


def test_load_wav_checks_format(tmp_path):
    good = tmp_path / "good.wav"
    wavfile.write(good, FS, tone(3.0, offset=0.03).astype(np.float32))
    x = load_wav(good)
    assert x.dtype == np.float64 and len(x) == L

    wrong_rate = tmp_path / "rate.wav"
    wavfile.write(wrong_rate, 16000, tone(3.0).astype(np.float32))
    with pytest.raises(ValueError, match="sample rate"):
        load_wav(wrong_rate)

    stereo = tmp_path / "stereo.wav"
    wavfile.write(stereo, FS, np.stack([tone(3.0)] * 2, axis=1).astype(np.float32))
    with pytest.raises(ValueError, match="channels"):
        load_wav(stereo)


def test_every_real_recording_gives_segments():
    """D-009: every DeepShip WAV gives at least one segment; totals as in D-015."""
    if not DATA_DIR.is_dir() or not any(DATA_DIR.glob("*/*.wav")):
        pytest.skip(
            f"{DATA_DIR} not found or without WAVs: download DeepShip (README.md)"
        )
    with SPLIT.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    counts = Counter()
    for r in rows:
        n = len(preprocess(load_wav(DATA_DIR / r["wav_path"])))
        assert n >= 1, r["wav_path"]
        counts[r["vessel_class"]] += n
    assert sum(counts.values()) == 1889
    assert counts == {"Cargo": 771, "Passengership": 377, "Tanker": 545, "Tug": 196}
