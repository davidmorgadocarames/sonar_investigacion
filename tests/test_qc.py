"""QC tests with synthetic signals whose answer is known in advance."""

import csv
import hashlib

import numpy as np
import pytest
from scipy.io import wavfile

from sonar_pipeline.__main__ import main
from sonar_pipeline.qc import analyze, natural_key, run_qc

FS = 32000


def tone(seconds=2.0, freq=100.0, amplitude=0.5, offset=0.0, dtype=np.float32):
    t = np.arange(int(seconds * FS)) / FS
    return (offset + amplitude * np.sin(2 * np.pi * freq * t)).astype(dtype)


def test_clean_tone_has_no_flags():
    r = analyze(tone(), FS)
    assert r["duration_s"] == pytest.approx(2.0)
    assert r["dc_offset"] == pytest.approx(0.0, abs=1e-6)
    assert r["std"] == pytest.approx(0.5 / np.sqrt(2), rel=1e-3)
    assert r["peak"] == pytest.approx(0.5, rel=1e-3)
    assert not any(v for k, v in r.items() if k.startswith("flag_"))


def test_known_dc_offset_is_measured_and_flagged():
    r = analyze(tone(offset=0.2), FS)
    assert r["dc_offset"] == pytest.approx(0.2, abs=1e-4)
    assert r["flag_dc"]


def test_small_dc_offset_below_threshold_is_not_flagged():
    # 0.01 / (0.5 / sqrt(2)) ~ 0.028 < 0.1
    r = analyze(tone(offset=0.01), FS)
    assert r["dc_offset"] == pytest.approx(0.01, abs=1e-4)
    assert not r["flag_dc"]


def test_clipping_is_counted_for_float():
    x = np.clip(tone(amplitude=2.0), -1.0, 1.0)
    r = analyze(x, FS)
    assert r["n_clipped"] > 0
    assert r["peak"] == pytest.approx(1.0)
    assert r["flag_clipping"]


def test_clipping_is_counted_for_int16():
    x = np.clip(tone(amplitude=2.0, dtype=np.float64) * 32768, -32768, 32767).astype(
        np.int16
    )
    r = analyze(x, FS)
    assert r["flag_clipping"]
    assert r["flag_dtype"]


def test_nonfinite_samples_are_counted_and_excluded_from_stats():
    x = tone()
    x[[10, 20, 30]] = [np.nan, np.inf, -np.inf]
    r = analyze(x, FS)
    assert r["n_nonfinite"] == 3
    assert r["flag_nonfinite"]
    assert np.isfinite(r["std"])


def test_silence_is_flagged():
    r = analyze(np.zeros(FS, dtype=np.float32), FS)
    assert r["flag_silent"]
    assert not r["flag_dc"]


def test_format_mismatch_is_flagged():
    stereo = np.stack([tone(), tone()], axis=1)
    r = analyze(stereo, 16000)
    assert r["channels"] == 2
    assert r["flag_channels"]
    assert r["flag_sample_rate"]
    assert r["duration_s"] == pytest.approx(4.0)


def test_run_qc_writes_one_row_per_wav(tmp_path):
    data = tmp_path / "data"
    (data / "Cargo").mkdir(parents=True)
    (data / "Tug").mkdir()
    wavfile.write(data / "Cargo" / "1.wav", FS, tone(seconds=3.0))
    wavfile.write(data / "Cargo" / "10.wav", FS, tone())
    wavfile.write(data / "Tug" / "2.wav", FS, tone(offset=0.3))
    out = tmp_path / "out" / "qc.csv"
    before = digests(data)

    results = run_qc(data, out)

    assert b"\r" not in out.read_bytes()
    rows = list(csv.DictReader(out.open(newline="", encoding="utf-8")))
    assert [r["path"] for r in rows] == ["Cargo/1.wav", "Cargo/10.wav", "Tug/2.wav"]
    assert [r["vessel_class"] for r in rows] == ["Cargo", "Cargo", "Tug"]
    assert float(rows[0]["duration_s"]) == pytest.approx(3.0)
    assert rows[0]["dtype"] == "float32"
    assert rows[0]["flagged"] == "False"
    assert rows[2]["flag_dc"] == "True"
    assert len(results) == 3
    # QC never touches the data: same files, same bytes.
    assert digests(data) == before


def digests(folder):
    return {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.rglob("*")
        if p.is_file()
    }


def test_natural_key_orders_numbers_by_value():
    paths = ["Tug/2.wav", "Cargo/10.wav", "Cargo/9.wav", "Cargo/103.wav"]
    assert sorted(paths, key=natural_key) == [
        "Cargo/9.wav",
        "Cargo/10.wav",
        "Cargo/103.wav",
        "Tug/2.wav",
    ]


def test_run_qc_fails_without_wavs(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_qc(tmp_path, tmp_path / "qc.csv")


def test_cli_qc(tmp_path, capsys):
    (tmp_path / "Cargo").mkdir()
    wavfile.write(tmp_path / "Cargo" / "1.wav", FS, tone())
    out = tmp_path / "qc.csv"
    assert main(["qc", "--data-dir", str(tmp_path), "--output", str(out)]) == 0
    assert "1 WAV files checked, 0 flagged" in capsys.readouterr().out
    assert out.exists()
