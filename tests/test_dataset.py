"""Model data tests (F3-T03): toy features with a known answer, plus the real ones if present."""

import csv
from pathlib import Path

import numpy as np
import pytest

from sonar_pipeline.dataset import (
    CLASSES,
    aggregate_by_recording,
    baseline_features,
    class_weights,
    load_split,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "datasets" / "processed"
FIELDS = ["segment_id", "wav_path", "vessel_class", "vessel_group", "split"]

# segment_id, wav_path, class, split
TOY = [
    (0, "Cargo/1.wav", "Cargo", "train"),
    (1, "Cargo/1.wav", "Cargo", "train"),
    (2, "Tug/2.wav", "Tug", "val"),
    (3, "Tanker/3.wav", "Tanker", "train"),
    (4, "Passengership/4.wav", "Passengership", "test"),
]


def write_toy(folder, rows=TOY, npz_ids=None):
    """index.csv plus one .npz per split; LOFAR value = segment id, to trace every row."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(FIELDS)
        for seg, path, cls, split in rows:
            writer.writerow([seg, path, cls, f"{cls}/V{seg}", split])
    for split in ("train", "val", "test"):
        ids = (npz_ids or {}).get(split, [r[0] for r in rows if r[3] == split])
        np.savez(
            folder / f"features_{split}.npz",
            segment_id=np.array(ids, dtype=np.int64),
            lofar=np.array([np.full((2, 3), i, np.float32) for i in ids]).reshape(
                -1, 2, 3
            ),
            demon=np.array([np.full(4, -i, np.float32) for i in ids]).reshape(-1, 4),
            lofar_hz=np.arange(3.0),
            demon_hz=np.arange(4.0),
        )
    return folder


def test_load_split_keeps_rows_aligned(tmp_path):
    data = load_split(write_toy(tmp_path), "train")
    assert len(data) == 3
    assert data.segment_id.tolist() == [0, 1, 3]
    assert data.labels.tolist() == [CLASSES.index("Cargo")] * 2 + [
        CLASSES.index("Tanker")
    ]
    assert data.wav_path.tolist() == ["Cargo/1.wav", "Cargo/1.wav", "Tanker/3.wav"]
    # Every array row belongs to its segment id.
    assert [float(x) for x in data.lofar[:, 0, 0]] == [0.0, 1.0, 3.0]
    assert [float(x) for x in data.demon[:, 0]] == [0.0, -1.0, -3.0]


def test_a_segment_of_another_split_is_rejected(tmp_path):
    # Segment 2 is val in index.csv but sits in features_train.npz.
    folder = write_toy(tmp_path, npz_ids={"train": [0, 1, 2, 3]})
    with pytest.raises(ValueError, match="segments of the \\['val'\\] split"):
        load_split(folder, "train")


def test_a_missing_segment_is_rejected(tmp_path):
    folder = write_toy(tmp_path, npz_ids={"train": [0, 1]})
    with pytest.raises(ValueError, match="disagree"):
        load_split(folder, "train")


def test_unknown_split_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown split"):
        load_split(write_toy(tmp_path), "holdout")


def test_class_weights_are_balanced():
    labels = np.array([0, 0, 0, 0, 1, 1, 2, 3])
    weights = class_weights(labels)
    # n / (4 * count): 8/16, 8/8, 8/4, 8/4
    np.testing.assert_allclose(weights, [0.5, 1.0, 2.0, 2.0])
    # Every class then weighs the same in total.
    np.testing.assert_allclose(weights * np.bincount(labels), 2.0)


def test_class_weights_need_every_class():
    with pytest.raises(ValueError, match="Tug"):
        class_weights(np.array([0, 1, 2]))


def test_baseline_features_are_time_mean_lofar_and_demon(tmp_path):
    data = load_split(write_toy(tmp_path), "train")
    x = baseline_features(data)
    assert x.shape == (3, 3 + 4)
    np.testing.assert_array_equal(x[2], [3, 3, 3, -3, -3, -3, -3])


def test_aggregate_by_recording_averages_probabilities():
    probs = np.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8], [0.4, 0.6], [0.0, 1.0]])
    paths = np.array(["a.wav", "a.wav", "b.wav", "b.wav", "c.wav"])
    labels = np.array([0, 0, 1, 1, 1])
    rec = aggregate_by_recording(probs, paths, labels)
    assert rec.wav_path.tolist() == ["a.wav", "b.wav", "c.wav"]
    np.testing.assert_allclose(rec.probabilities, [[0.7, 0.3], [0.3, 0.7], [0.0, 1.0]])
    assert rec.labels.tolist() == [0, 1, 1]


def test_aggregate_rejects_mixed_labels_in_one_recording():
    with pytest.raises(ValueError, match="different labels"):
        aggregate_by_recording(
            np.ones((2, 2)) / 2, np.array(["a.wav", "a.wav"]), np.array([0, 1])
        )


def test_real_features_load_without_mixing_splits():
    if not (PROCESSED / "index.csv").is_file():
        pytest.skip(f"{PROCESSED} has no features: run `features` first (README)")
    splits = {s: load_split(PROCESSED, s) for s in ("train", "val", "test")}
    assert {s: len(d) for s, d in splits.items()} == {
        "train": 1077,
        "val": 380,
        "test": 432,
    }
    for s, d in splits.items():
        assert d.lofar.shape[1:] == (11, 1001) and d.demon.shape[1:] == (301,)
        assert np.all(np.bincount(d.labels, minlength=len(CLASSES)) > 0), (
            s
        )  # every class
    # No recording and no vessel in two splits.
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(splits[a].wav_path) & set(splits[b].wav_path)
        assert not set(splits[a].vessel_group) & set(splits[b].vessel_group)


def test_eras_match_the_recording_dates():
    import csv as _csv

    from sonar_pipeline.dataset import ERAS

    vessels = ROOT / "datasets" / "manifests" / "vessels.csv"
    if not vessels.is_file():
        pytest.skip(
            f"{vessels} not found: datasets/ is not mounted (use docker compose)"
        )
    years = {}
    with vessels.open(newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            years.setdefault(r["vessel_class"], set()).add(r["date"][:4])
    assert {cls: era for era, pair in ERAS.items() for cls in pair} == {
        cls: next(iter(ys)) for cls, ys in years.items() if len(ys) == 1
    }
    assert sorted(c for pair in ERAS.values() for c in pair) == sorted(CLASSES)
