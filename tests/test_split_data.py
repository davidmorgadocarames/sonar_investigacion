"""Permanent checks on the versioned split manifest (F1-T06c, D-012, D-013).

``datasets/manifests/split.csv`` and ``vessels.csv`` are versioned, so these tests always run:
the split must have no leaks, every class must have a group in every split (Tug 1/1/1), and
regenerating it with tools/make_split.py must give the same bytes.
"""

import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "datasets" / "manifests"
SPLIT = MANIFESTS / "split.csv"
VESSELS = MANIFESTS / "vessels.csv"
SPLITS = ("train", "val", "test")


def load_tool(name):
    spec = importlib.util.spec_from_file_location(
        f"{name}_data", ROOT / "tools" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_leaks = load_tool("check_leaks")
make_split = load_tool("make_split")


def read(path):
    assert path.is_file(), f"{path} is versioned and must exist"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def split_rows():
    return read(SPLIT)


@pytest.fixture(scope="module")
def vessel_rows():
    return read(VESSELS)


def test_versioned_split_has_no_leaks(split_rows, vessel_rows):
    assert check_leaks.check_split(split_rows, vessel_rows) == []


def test_format_and_order_match_vessels(split_rows, vessel_rows):
    raw = SPLIT.read_bytes()
    assert b"\r" not in raw
    raw.decode("utf-8")
    assert raw.splitlines()[0] == b"wav_path,vessel_class,vessel_group,split"
    assert [r["wav_path"] for r in split_rows] == [r["wav_path"] for r in vessel_rows]
    assert len(split_rows) == 63


def test_every_class_has_a_group_in_every_split(split_rows):
    groups = defaultdict(set)
    for r in split_rows:
        groups[(r["vessel_class"], r["split"])].add(r["vessel_group"])
    classes = {r["vessel_class"] for r in split_rows}
    assert classes == {"Cargo", "Passengership", "Tanker", "Tug"}
    for c in classes:
        for s in SPLITS:
            assert groups[(c, s)], f"{c} has no group in {s}"
    assert [len(groups[("Tug", s)]) for s in SPLITS] == [1, 1, 1]


def test_regenerating_gives_identical_bytes(tmp_path):
    out = tmp_path / "split.csv"
    assert make_split.main(["--vessels", str(VESSELS), "--output", str(out)]) == 0
    assert out.read_bytes() == SPLIT.read_bytes()
