"""Permanent checks on the versioned cross-validation partitions (F5-T02, D-024).

``datasets/manifests/cv_splits.csv`` and ``vessels.csv`` are versioned, so these tests run
whenever ``datasets/`` is there (always in a clone and with docker compose; skipped with a bare
``docker run``). They check the 25 partitions independently of the generator's own checks:
no leaks (``check_split`` of tools/check_leaks.py on each partition); every vessel of the
classes with 5 or more vessels once in test and once in validation per repetition; every class in
train, val and test of every partition, with Tug exactly 1/1/1 and each Tug vessel in test and in
validation 8 or 9 times out of 25; balanced minutes; different repetitions; and same bytes on
regeneration.
"""

import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "datasets" / "manifests"
CV_SPLITS = MANIFESTS / "cv_splits.csv"
VESSELS = MANIFESTS / "vessels.csv"
CLASSES = {"Cargo", "Passengership", "Tanker", "Tug"}
BIG = {"Cargo", "Passengership", "Tanker"}


def load_tool(name):
    spec = importlib.util.spec_from_file_location(
        f"{name}_cvdata", ROOT / "tools" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_leaks = load_tool("check_leaks")
make_cv_splits = load_tool("make_cv_splits")


def read(path):
    if not MANIFESTS.is_dir():
        pytest.skip(
            f"{MANIFESTS} not found: datasets/ is not mounted (use docker compose)"
        )
    assert path.is_file(), f"{path} is versioned and must exist"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def cv_rows():
    return read(CV_SPLITS)


@pytest.fixture(scope="module")
def vessel_rows():
    return read(VESSELS)


@pytest.fixture(scope="module")
def parts(cv_rows):
    out = defaultdict(list)
    for r in cv_rows:
        out[(int(r["repetition"]), int(r["fold"]))].append(r)
    return dict(out)


def test_format_and_order(cv_rows, vessel_rows, parts):
    raw = CV_SPLITS.read_bytes()
    assert b"\r" not in raw
    raw.decode("utf-8")
    assert (
        raw.splitlines()[0]
        == b"repetition,fold,wav_path,vessel_class,vessel_group,split"
    )
    assert len(vessel_rows) == 63
    assert len(cv_rows) == 25 * 63 == 1575
    keys = [(int(r["repetition"]), int(r["fold"])) for r in cv_rows]
    assert keys == sorted(keys)
    assert set(parts) == {(r, k) for r in range(5) for k in range(5)}
    for part in parts.values():
        assert [p["wav_path"] for p in part] == [v["wav_path"] for v in vessel_rows]


def test_no_leaks_in_any_of_the_25_partitions(parts, vessel_rows):
    for key, part in sorted(parts.items()):
        assert check_leaks.check_split(part, vessel_rows) == [], key


def test_big_class_vessels_exactly_once_in_test_and_once_in_val(parts, vessel_rows):
    all_groups = {v["vessel_group"] for v in vessel_rows}
    assert len(all_groups) == 43
    big_groups = {v["vessel_group"] for v in vessel_rows if v["vessel_class"] in BIG}
    assert len(big_groups) == 40
    for r in range(5):
        for s in ("test", "val"):
            folds = defaultdict(set)
            for k in range(5):
                for p in parts[(r, k)]:
                    if p["split"] == s and p["vessel_class"] in BIG:
                        folds[p["vessel_group"]].add(k)
            assert set(folds) == big_groups, (r, s)
            assert all(len(f) == 1 for f in folds.values()), (r, s)


def test_every_class_in_train_val_and_test_in_every_partition(parts):
    for key, part in parts.items():
        present = defaultdict(set)
        for p in part:
            present[(p["vessel_class"], p["split"])].add(p["vessel_group"])
        for c in CLASSES:
            for s in ("train", "val", "test"):
                assert present[(c, s)], (key, c, s)
        assert [len(present[("Tug", s)]) for s in ("train", "val", "test")] == [1, 1, 1]


def test_tug_vessels_balanced_in_test_and_val(cv_rows, vessel_rows):
    tugs = sorted(
        {v["vessel_group"] for v in vessel_rows if v["vessel_class"] == "Tug"}
    )
    assert len(tugs) == 3
    counts = make_cv_splits.role_counts(cv_rows)
    for s in ("test", "val", "train"):
        got = sorted(counts.get((g, s), 0) for g in tugs)
        assert got == [8, 8, 9], (s, got)


def test_big_class_val_and_test_folds_are_consecutive(parts):
    # Within a repetition the test fold of round k is the validation fold of round k - 1.
    for r in range(5):
        for k in range(5):
            test = {
                p["vessel_group"]
                for p in parts[(r, k)]
                if p["split"] == "test" and p["vessel_class"] in BIG
            }
            val_prev = {
                p["vessel_group"]
                for p in parts[(r, (k - 1) % 5)]
                if p["split"] == "val" and p["vessel_class"] in BIG
            }
            assert test == val_prev, (r, k)


def test_minutes_per_class_are_balanced_across_folds(cv_rows, vessel_rows):
    # Bounds measured on the versioned file (granularity of the vessels makes perfect balance
    # impossible for Cargo and Passengership; Tug does not use folds).
    bounds = {"Cargo": 0.30, "Passengership": 0.45, "Tanker": 0.15}
    minutes = make_cv_splits.fold_minutes(cv_rows, vessel_rows)
    for (r, cls), mins in minutes.items():
        if cls not in bounds:
            continue
        mean = sum(mins) / 5
        assert all(m > 0 for m in mins), (r, cls)
        assert max(abs(m - mean) for m in mins) / mean <= bounds[cls], (r, cls, mins)


def test_repetitions_are_different(cv_rows):
    groupings = make_cv_splits.class_partitions(cv_rows)
    for cls in ("Cargo", "Passengership", "Tanker"):
        assert len({groupings[(r, cls)] for r in range(5)}) == 5, cls
    # And the labelled assignment of the whole table differs between every pair.
    fold_of = defaultdict(dict)
    for p in cv_rows:
        if p["split"] == "test" and p["vessel_class"] in BIG:
            fold_of[int(p["repetition"])][p["vessel_group"]] = int(p["fold"])
    assert len({tuple(sorted(fold_of[r].items())) for r in range(5)}) == 5


def test_tug_pairs_do_not_repeat_within_a_repetition(parts):
    for r in range(5):
        pairs = []
        for k in range(5):
            role = {
                p["split"]: p["vessel_group"]
                for p in parts[(r, k)]
                if p["vessel_class"] == "Tug" and p["split"] != "train"
            }
            pairs.append((role["test"], role["val"]))
        assert len(set(pairs)) == 5, (r, pairs)


def test_regeneration_gives_the_same_bytes(tmp_path, capsys):
    out = tmp_path / "cv_splits.csv"
    assert make_cv_splits.main(["--vessels", str(VESSELS), "--output", str(out)]) == 0
    assert out.read_bytes() == CV_SPLITS.read_bytes()
