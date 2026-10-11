"""Figures (F2-T08, D-016). The validation figures show synthetic signals with a known answer.

To keep them (synthetic signals only exist in tests/, see CLAUDE.md):

    docker compose run --rm --entrypoint python pipeline -m pytest tests/test_figures.py \
        --figures-dir outputs/figures
"""

import numpy as np
import pytest

from sonar_pipeline.demon import demon, demon_frequencies
from sonar_pipeline.figures import (
    lofar_time_grid,
    plot_recording,
    plot_spectrum_with_marker,
)
from sonar_pipeline.lofar import lofar, lofar_frequencies
from sonar_pipeline.params import SAMPLE_RATE
from sonar_pipeline.preprocess import preprocess

FS = SAMPLE_RATE
N = 3 * FS
T = np.arange(N) / FS
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
TONE_HZ = 440.0
MODULATION_HZ = 12.0


@pytest.fixture
def figures_dir(request, tmp_path):
    return request.config.getoption("--figures-dir") or tmp_path


def noise(seed=0):
    return np.random.default_rng(seed).standard_normal(N)


def assert_png(path):
    assert path.is_file()
    assert path.read_bytes()[:8] == PNG_SIGNATURE
    assert path.stat().st_size > 10_000


def test_validation_lofar_tone(figures_dir):
    gram = lofar(preprocess(0.1 * np.sin(2 * np.pi * TONE_HZ * T) + noise())[0])
    spectrum = gram.mean(axis=0)
    hz = lofar_frequencies()
    assert abs(hz[np.argmax(spectrum)] - TONE_HZ) <= 2.0
    path = plot_spectrum_with_marker(
        hz,
        spectrum,
        TONE_HZ,
        "Validación LOFAR: tono de 440 Hz (amplitud 0,1) en ruido blanco (amplitud 1)",
        "Frecuencia (Hz)",
        "dB sobre el fondo (media de 11 tramas)",
        figures_dir / "validation_lofar.png",
    )
    assert_png(path)


def test_validation_demon_modulation(figures_dir):
    x = (1 + 0.5 * np.cos(2 * np.pi * MODULATION_HZ * T)) * noise(1)
    spectrum = demon(preprocess(x)[0])
    hz = demon_frequencies(N)
    assert abs(hz[np.argmax(spectrum)] - MODULATION_HZ) <= 1 / 3
    kept = hz >= 1.0
    path = plot_spectrum_with_marker(
        hz[kept],
        spectrum[kept],
        MODULATION_HZ,
        "Validación DEMON: ruido blanco modulado en amplitud a 12 Hz (profundidad 50 %)",
        "Frecuencia de modulación (Hz)",
        "Amplitud relativa",
        figures_dir / "validation_demon.png",
    )
    assert_png(path)


def test_plot_recording_writes_a_png(tmp_path):
    segments = preprocess(np.concatenate([noise(2), noise(3)]))
    path = plot_recording(
        np.stack([lofar(s) for s in segments]),
        lofar_frequencies(),
        demon(segments[0]),
        demon_frequencies(N),
        "Prueba",
        tmp_path / "sub" / "example.png",
    )
    assert_png(path)


def test_lofar_time_grid_places_frames_at_their_true_time():
    # 2 segments x 11 frames: frame centres 0.25..2.75 s and 3.25..5.75 s, one empty row
    # at 3.0 s between them, every row 0.25 s high.
    segments = np.arange(2 * 11 * 3, dtype=float).reshape(2, 11, 3)
    rows, (top, bottom) = lofar_time_grid(segments)
    assert rows.shape == (23, 3)
    np.testing.assert_array_equal(rows[:11], segments[0])
    assert np.all(np.isnan(rows[11]))
    np.testing.assert_array_equal(rows[12:], segments[1])
    assert top == pytest.approx(0.125)
    assert bottom == pytest.approx(5.875)
    centres = top + 0.25 * (np.arange(23) + 0.5)
    assert centres[0] == pytest.approx(0.25) and centres[12] == pytest.approx(3.25)


def test_data_minutes_from_the_manifests(tmp_path):
    import csv

    from sonar_pipeline.figures import data_minutes, plot_data_minutes

    vessels = tmp_path / "vessels.csv"
    split = tmp_path / "split.csv"
    rows = [
        ("Cargo/1.wav", "Cargo", "Cargo/A", "train", 120.0),
        ("Cargo/2.wav", "Cargo", "Cargo/A", "train", 60.0),
        ("Cargo/3.wav", "Cargo", "Cargo/B", "test", 30.0),
        ("Tug/4.wav", "Tug", "Tug/C", "val", 90.0),
    ]
    with vessels.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["wav_path", "wav_duration_s"])
        w.writerows((r[0], r[4]) for r in rows)
    with split.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["wav_path", "vessel_class", "vessel_group", "split"])
        w.writerows(r[:4] for r in rows)
    table = data_minutes(vessels, split)
    assert table["Cargo"]["train"] == {"minutes": 3.0, "recordings": 2, "vessels": 1}
    assert table["Cargo"]["test"] == {"minutes": 0.5, "recordings": 1, "vessels": 1}
    assert table["Tug"]["val"]["minutes"] == 1.5
    assert_png(plot_data_minutes(vessels, split, tmp_path / "data.png"))


def test_data_minutes_of_the_real_manifests():
    from pathlib import Path

    from sonar_pipeline.figures import data_minutes

    manifests = Path(__file__).resolve().parents[1] / "datasets" / "manifests"
    if not manifests.is_dir():
        pytest.skip(
            f"{manifests} not found: datasets/ is not mounted (use docker compose)"
        )
    table = data_minutes(manifests / "vessels.csv", manifests / "split.csv")
    total = sum(
        cell["minutes"] for splits in table.values() for cell in splits.values()
    )
    assert total == pytest.approx(95.4, abs=0.1)  # D-013
    assert [table["Tug"][s]["vessels"] for s in ("train", "val", "test")] == [1, 1, 1]
    assert (
        sum(cell["vessels"] for splits in table.values() for cell in splits.values())
        == 43
    )
