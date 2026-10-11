"""Tests for the split algorithm of tools/make_split.py with synthetic vessel groups."""

import csv
import importlib.util
import itertools
import random
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "make_split.py"
spec = importlib.util.spec_from_file_location("make_split", TOOL)
ms = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ms
spec.loader.exec_module(ms)


def groups(*durations_s, prefix="G"):
    return [
        ms.Group(f"{prefix}{i:02d}", seconds * 10**6, 1)
        for i, seconds in enumerate(durations_s)
    ]


def split_sums(groups_, assignment):
    sums = dict.fromkeys(ms.SPLITS, 0)
    for g in groups_:
        sums[assignment[g.name]] += g.duration_us
    return tuple(sums[s] for s in ms.SPLITS)


def brute_force(groups_, require_all_splits=True):
    """Minimal scaled cost over all 3^n assignments, in the search order."""
    order = ms.search_order(groups_)
    total = sum(g.duration_us for g in order)
    best = None
    for combo in itertools.product(range(3), repeat=len(order)):
        if require_all_splits and len(set(combo)) < 3:
            continue
        sums = [0, 0, 0]
        for g, k in zip(order, combo):
            sums[k] += g.duration_us
        cost = ms.scaled_cost(tuple(sums), total)
        if best is None or cost < best[0]:  # first in lexicographic order wins ties
            best = (cost, {g.name: ms.SPLITS[k] for g, k in zip(order, combo)})
    return best


def test_scaled_cost_is_exact_l1_distance():
    assert ms.scaled_cost((60, 20, 20), 100) == 0
    # fractions 0.5/0.3/0.2 -> |0.5-0.6| + |0.3-0.2| + 0 = 0.2 -> 5 * 100 * 0.2 = 100
    assert ms.scaled_cost((50, 30, 20), 100) == 100


def test_duration_us_is_exact():
    assert ms.duration_us("187.000000") == 187_000_000
    assert ms.duration_us("0.000001") == 1
    with pytest.raises(ms.SplitError):
        ms.duration_us("0.0000001")


def test_known_optimum_is_exact_60_20_20():
    g = groups(30, 30, 20, 20)
    result = ms.assign_class(g)
    assert split_sums(g, result) == (60 * 10**6, 20 * 10**6, 20 * 10**6)
    # Tie between the two 20 s groups: the first by name goes to val (documented order).
    assert result == {"G00": "train", "G01": "train", "G02": "val", "G03": "test"}


def test_every_split_gets_a_group_even_if_cost_is_higher():
    g = groups(10, 10, 10)
    # Without the constraint the optimum is 20/10/0 (cost 0.4), leaving test empty...
    unconstrained_cost, unconstrained = brute_force(g, require_all_splits=False)
    assert "test" not in unconstrained.values()
    # ...with it, every split gets a group: 10/10/10 (cost 0.533).
    result = ms.assign_class(g)
    assert set(result.values()) == set(ms.SPLITS)
    cost, expected = brute_force(g)
    assert cost > unconstrained_cost
    assert result == expected


def test_class_with_exactly_three_groups_is_one_per_split():
    g = groups(190, 203, 200)
    result = ms.assign_class(g)
    # The largest group goes to train; the other two tie on cost and follow the order.
    assert result == {"G01": "train", "G02": "val", "G00": "test"}


def test_fewer_than_three_groups_fails():
    with pytest.raises(ms.SplitError, match="need at least 3"):
        ms.assign_class(groups(10, 20))


def test_matches_brute_force_on_random_instances():
    rng = random.Random(
        13
    )  # synthetic test data only; the tool itself has no randomness
    for _ in range(40):
        n = rng.randint(3, 8)
        g = groups(*(rng.randint(1, 250) for _ in range(n)))
        cost, expected = brute_force(g)
        result = ms.assign_class(g)
        assert result == expected
        total = sum(x.duration_us for x in g)
        assert ms.scaled_cost(split_sums(g, result), total) == cost


def test_assignment_does_not_depend_on_input_order():
    g = groups(30, 7, 22, 15, 15, 9, 41)
    assert ms.assign_class(g) == ms.assign_class(list(reversed(g)))


# --------------------------------------------------------------------------- CLI


def write_vessels(path, rows):
    fields = ["wav_path", "vessel_class", "vessel_group", "sha256", "wav_duration_s"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for i, (path_, group, seconds) in enumerate(rows):
            writer.writerow(
                {
                    "wav_path": path_,
                    "vessel_class": path_.split("/")[0],
                    "vessel_group": group,
                    "sha256": f"{i:064x}",
                    "wav_duration_s": f"{seconds:.6f}",
                }
            )
    return path


def test_main_writes_split_in_vessels_order(tmp_path, capsys):
    vessels = write_vessels(
        tmp_path / "vessels.csv",
        [
            ("Tug/1.wav", "Tug/A", 100),
            ("Tug/2.wav", "Tug/B", 300),
            ("Tug/3.wav", "Tug/C", 100),
            ("Tug/4.wav", "Tug/B", 300),
        ],
    )
    out = tmp_path / "split.csv"
    assert ms.main(["--vessels", str(vessels), "--output", str(out)]) == 0
    assert out.read_bytes() == (
        b"wav_path,vessel_class,vessel_group,split\n"
        b"Tug/1.wav,Tug,Tug/A,val\n"
        b"Tug/2.wav,Tug,Tug/B,train\n"
        b"Tug/3.wav,Tug,Tug/C,test\n"
        b"Tug/4.wav,Tug,Tug/B,train\n"
    )
    assert "cost = sum|frac - target|" in capsys.readouterr().out


def test_main_fails_and_writes_nothing_with_too_few_groups(tmp_path, capsys):
    vessels = write_vessels(
        tmp_path / "vessels.csv",
        [("Tug/1.wav", "Tug/A", 100), ("Tug/2.wav", "Tug/B", 100)],
    )
    out = tmp_path / "split.csv"
    assert ms.main(["--vessels", str(vessels), "--output", str(out)]) == 1
    assert not out.exists()
    assert "class Tug" in capsys.readouterr().err


def test_main_writes_nothing_if_leak_check_fails(tmp_path, monkeypatch, capsys):
    vessels = write_vessels(
        tmp_path / "vessels.csv",
        [
            ("Tug/1.wav", "Tug/A", 1),
            ("Tug/2.wav", "Tug/B", 1),
            ("Tug/3.wav", "Tug/C", 1),
        ],
    )
    monkeypatch.setattr(ms.check_leaks, "check_split", lambda s, v: ["vessel leak: X"])
    out = tmp_path / "split.csv"
    assert ms.main(["--vessels", str(vessels), "--output", str(out)]) == 1
    assert not out.exists()
    assert "vessel leak: X" in capsys.readouterr().err
