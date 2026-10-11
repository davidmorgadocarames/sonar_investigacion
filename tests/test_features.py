"""End-to-end test of the features subcommand (F2-T07) on a small synthetic dataset."""

import csv
import hashlib
import json

import numpy as np
import pytest
from scipy.io import wavfile

from sonar_pipeline.__main__ import main
from sonar_pipeline.params import SAMPLE_RATE

FS = SAMPLE_RATE

# wav_path, class, group, split, seconds, tone (Hz), modulation (Hz)
RECORDINGS = [
    ("Cargo/1.wav", "Cargo", "Cargo/A", "train", 7.0, 440.0, 12.0),
    ("Tug/5.wav", "Tug", "Tug/B", "val", 3.5, 1000.0, 7.0),
    ("Tanker/2.wav", "Tanker", "Tanker/C", "test", 9.2, 250.0, 30.0),
    ("Cargo/3.wav", "Cargo", "Cargo/D", "train", 6.0, 600.0, 5.0),
]
EXPECTED_SEGMENTS = {
    "Cargo/1.wav": 2,
    "Tug/5.wav": 1,
    "Tanker/2.wav": 3,
    "Cargo/3.wav": 2,
}


def synthetic(seconds, tone_hz, mod_hz, seed):
    """DeepShip-like: DC offset, a tonal line and modulated broadband noise."""
    n = round(seconds * FS)
    t = np.arange(n) / FS
    rng = np.random.default_rng(seed)
    noise = (1 + 0.5 * np.cos(2 * np.pi * mod_hz * t)) * rng.standard_normal(n)
    return (0.03 + 0.005 * (np.sin(2 * np.pi * tone_hz * t) + 0.2 * noise)).astype(
        np.float32
    )


@pytest.fixture
def dataset(tmp_path):
    data = tmp_path / "raw"
    for seed, (path, *_rest, seconds, tone_hz, mod_hz) in enumerate(RECORDINGS):
        (data / path).parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(data / path, FS, synthetic(seconds, tone_hz, mod_hz, seed))
    split = tmp_path / "split.csv"
    with split.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["wav_path", "vessel_class", "vessel_group", "split"])
        writer.writerows(r[:4] for r in RECORDINGS)
    return data, split


def run(data, split, out):
    argv = ["features", "--data-dir", str(data), "--split", str(split)]
    return main([*argv, "--output-dir", str(out)])


def read_index(out):
    with (out / "index.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_features_end_to_end(dataset, tmp_path, capsys):
    data, split = dataset
    out = tmp_path / "processed"
    assert run(data, split, out) == 0
    assert "4 recordings, 8 segments" in capsys.readouterr().out

    index = read_index(out)
    assert b"\r" not in (out / "index.csv").read_bytes()
    assert [int(r["segment_id"]) for r in index] == list(range(8))
    # Rows follow split.csv order; segments of one recording are consecutive.
    order = [r["wav_path"] for r in index]
    assert order == [p for p, *_ in RECORDINGS for _ in range(EXPECTED_SEGMENTS[p])]
    for path, n in EXPECTED_SEGMENTS.items():
        rows = [r for r in index if r["wav_path"] == path]
        assert [int(r["segment_index"]) for r in rows] == list(range(n))
        assert [r["start_s"] for r in rows] == [f"{3.0 * i:.3f}" for i in range(n)]

    by_id = {int(r["segment_id"]): r for r in index}
    for split_name in ("train", "val", "test"):
        npz = np.load(out / f"features_{split_name}.npz")
        ids = npz["segment_id"]
        n = len(ids)
        assert {by_id[i]["split"] for i in ids} == {split_name}
        assert npz["lofar"].shape == (n, 11, 1001)
        assert npz["demon"].shape == (n, 301)
        assert npz["lofar"].dtype == npz["demon"].dtype == np.float32
        # Each segment shows its recording's tone in LOFAR and its modulation in DEMON.
        for k, segment_id in enumerate(ids):
            rec = next(r for r in RECORDINGS if r[0] == by_id[segment_id]["wav_path"])
            tone_hz, mod_hz = rec[5], rec[6]
            lofar_peak = npz["lofar_hz"][np.argmax(npz["lofar"][k].mean(axis=0))]
            demon_peak = npz["demon_hz"][np.argmax(npz["demon"][k])]
            assert abs(lofar_peak - tone_hz) <= 2.0
            assert abs(demon_peak - mod_hz) <= 1 / 3

    meta = json.loads((out / "features_meta.json").read_text(encoding="utf-8"))
    assert meta["decision"] == "D-015"
    assert meta["split_sha256"] == hashlib.sha256(split.read_bytes()).hexdigest()
    assert meta["params"]["SEGMENT_S"] == 3.0
    assert len(meta["code_sha256"]) == 64


def test_features_are_deterministic(dataset, tmp_path):
    data, split = dataset
    first, second = tmp_path / "first", tmp_path / "second"
    assert run(data, split, first) == 0
    assert run(data, split, second) == 0
    for name in ("index.csv", "features_meta.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    for split_name in ("train", "val", "test"):
        a = np.load(first / f"features_{split_name}.npz")
        b = np.load(second / f"features_{split_name}.npz")
        assert a.files == b.files
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])


def test_missing_wav_fails(dataset, tmp_path):
    data, split = dataset
    (data / "Tug" / "5.wav").unlink()
    with pytest.raises(FileNotFoundError, match="Tug/5.wav"):
        run(data, split, tmp_path / "out")


def test_unknown_split_value_fails(dataset, tmp_path):
    data, split = dataset
    split.write_text(
        split.read_text(encoding="utf-8").replace(",val\n", ",holdout\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="holdout"):
        run(data, split, tmp_path / "out")
