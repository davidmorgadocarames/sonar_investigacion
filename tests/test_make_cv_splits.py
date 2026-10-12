"""Tests for tools/make_cv_splits.py (F5-T02, D-024) with synthetic vessel groups."""

import csv
import importlib.util
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "make_cv_splits.py"
spec = importlib.util.spec_from_file_location("make_cv_splits", TOOL)
cv = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cv
spec.loader.exec_module(cv)


def groups(*durations_s, prefix="G"):
    return [
        cv.Group(f"{prefix}{i:02d}", seconds * 10**6, 1)
        for i, seconds in enumerate(durations_s)
    ]


def fold_sums(groups_, folds):
    sums = [0] * cv.N_FOLDS
    for g in groups_:
        sums[folds[g.name]] += g.duration_us
    return sums


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


# A small synthetic dataset: Tug with 3 vessels (as in DeepShip), Four with 4, A with 6, B with 9
# (one vessel with two recordings).
SYNTHETIC = (
    [
        ("Tug/1.wav", "Tug/T1", 200),
        ("Tug/2.wav", "Tug/T2", 190),
        ("Tug/3.wav", "Tug/T3", 210),
    ]
    + [(f"Four/{i}.wav", f"Four/F{i}", 150 + 11 * i) for i in range(4)]
    + [(f"A/{i}.wav", f"A/V{i}", 100 + 37 * i) for i in range(6)]
    + [(f"B/{i}.wav", f"B/W{i}", 50 + 23 * i) for i in range(9)]
    + [("B/9.wav", "B/W0", 80)]
)
SMALL = {
    "Tug": ["Tug/T1", "Tug/T2", "Tug/T3"],
    "Four": ["Four/F0", "Four/F1", "Four/F2", "Four/F3"],
}


def vessel_rows(rows=SYNTHETIC):
    return [
        {
            "wav_path": p,
            "vessel_class": p.split("/")[0],
            "vessel_group": g,
            "sha256": f"{i:064x}",
            "wav_duration_s": f"{s:.6f}",
        }
        for i, (p, g, s) in enumerate(rows)
    ]


# ------------------------------------------------------------- small classes (3 or 4 vessels)


def run_rotation(n, seed_base=0):
    """(test, val) pairs of the 25 partitions, as the tool draws them (seed = repetition)."""
    rot = cv.SmallClassRotation(groups(*([100] * n)))
    pairs = []
    for r in range(cv.N_REPETITIONS):
        rng = np.random.default_rng(seed_base + r)
        used = set()
        for _ in range(cv.N_FOLDS):
            pair = rot.next_pair(rng, used)
            used.add(pair)
            pairs.append(pair)
    return pairs


@pytest.mark.parametrize("n", [3, 4])
def test_rotation_balances_test_and_val_within_one_at_every_prefix(n):
    for base in range(0, 200, 10):
        pairs = run_rotation(n, base)
        assert len(pairs) == 25
        for end in range(1, 26):
            for side in (0, 1):
                counts = [sum(1 for p in pairs[:end] if p[side] == x) for x in range(n)]
                assert max(counts) - min(counts) <= 1, (base, end, side, counts)
        for side in (0, 1):
            final = sorted(sum(1 for p in pairs if p[side] == x) for x in range(n))
            assert final == ([8, 8, 9] if n == 3 else [6, 6, 6, 7])


@pytest.mark.parametrize("n", [3, 4])
def test_rotation_test_differs_from_val_and_no_pair_repeats_in_a_repetition(n):
    for base in range(0, 200, 10):
        pairs = run_rotation(n, base)
        assert all(t != v for t, v in pairs)
        for r in range(cv.N_REPETITIONS):
            block = pairs[5 * r : 5 * r + 5]
            assert len(set(block)) == 5, (base, r, block)


def test_rotation_is_deterministic_and_depends_on_the_seed():
    assert run_rotation(3, 0) == run_rotation(3, 0)
    assert len({tuple(run_rotation(3, base)) for base in range(0, 50, 10)}) > 1


def test_rotation_rejects_wrong_sizes():
    for n in (2, 5):
        with pytest.raises(cv.CVSplitError):
            cv.SmallClassRotation(groups(*([100] * n)))


def test_fewer_than_three_groups_fails():
    with pytest.raises(cv.CVSplitError, match="need at least 3"):
        cv.assign_roles({"Tug": groups(10, 20), "A": groups(*range(10, 70, 10))})


def test_duplicate_names_fail():
    g = groups(10, 20, 30) + [cv.Group("G00", 5 * 10**6, 1)]
    with pytest.raises(cv.CVSplitError, match="duplicate"):
        cv.assign_roles({"Tug": g})


def test_assign_folds_needs_five_groups():
    with pytest.raises(cv.CVSplitError, match="at least 5"):
        cv.assign_folds(groups(10, 20, 30, 40), np.random.default_rng(0))


# ------------------------------------------------------------------ fold assignment (n >= 5)


def test_five_or_more_groups_fill_every_fold():
    for n in (5, 6, 9, 13):
        for seed in range(5):
            g = groups(*(60 + 17 * i for i in range(n)))
            folds = cv.assign_folds(g, np.random.default_rng(seed))
            assert set(folds.values()) == set(range(cv.N_FOLDS))


def test_balance_is_within_the_tolerance():
    g = groups(400, 210, 205, 200, 198, 190, 120, 80, 45, 30, 12)
    durations = [x.duration_us for x in sorted(g, key=lambda x: x.name)]
    total = sum(durations)
    tolerance = cv.class_tolerance(durations)
    for seed in range(10):
        sums = fold_sums(g, cv.assign_folds(g, np.random.default_rng(seed)))
        assert cv.max_deviation(sums, total) <= tolerance


def test_reference_of_the_tolerance_is_the_exact_optimum_on_small_cases():
    # Brute force over all 5^n assignments with no empty fold: the local search reference
    # must reach the optimal max deviation on these small instances (6 and 7 groups).
    for durations_s in [(90, 70, 50, 40, 30, 20), (100, 33, 33, 33, 32, 31, 30)]:
        durations = [d * 10**6 for d in durations_s]
        total = sum(durations)
        best = min(
            cv.max_deviation(cv._sums(durations, list(c)), total)
            for c in itertools.product(range(cv.N_FOLDS), repeat=len(durations))
            if len(set(c)) == cv.N_FOLDS
        )
        assert cv.class_tolerance(durations) - total // 10 == best


def test_assignment_is_deterministic_and_independent_of_input_order():
    g = groups(30, 7, 22, 15, 15, 9, 41, 12)
    a = cv.assign_folds(g, np.random.default_rng(3))
    b = cv.assign_folds(list(reversed(g)), np.random.default_rng(3))
    assert a == b


def test_seeds_give_different_groupings():
    g = groups(*(100 + 7 * i for i in range(12)))

    def grouping(seed):
        folds = cv.assign_folds(g, np.random.default_rng(seed))
        by_fold = defaultdict(set)
        for name, f in folds.items():
            by_fold[f].add(name)
        return frozenset(frozenset(s) for s in by_fold.values())

    assert len({grouping(s) for s in range(5)}) == 5


def test_local_search_never_empties_a_fold():
    durations = [10, 10, 10, 10, 1000]
    folds = cv._local_search(durations, [0, 1, 2, 3, 4], sum(durations))
    assert sorted(folds) == [0, 1, 2, 3, 4]


def test_role_is_test_k_val_next_and_train_otherwise():
    assert [cv.role(f, 0) for f in range(5)] == [
        "test",
        "val",
        "train",
        "train",
        "train",
    ]
    assert [cv.role(f, 4) for f in range(5)] == [
        "val",
        "train",
        "train",
        "train",
        "test",
    ]


# ------------------------------------------------------------------ whole table


def test_synthetic_table_passes_all_checks():
    vessels = vessel_rows()
    rows = cv.make_cv_splits(vessels)
    assert len(rows) == 25 * len(vessels)
    assert cv.check_cv_splits(rows, vessels) == []
    # Order: repetition, fold, then vessels.csv order.
    keys = [(int(r["repetition"]), int(r["fold"])) for r in rows]
    assert keys == sorted(keys)
    for part in cv.partitions(rows).values():
        assert [p["wav_path"] for p in part] == [v["wav_path"] for v in vessels]


def test_synthetic_big_classes_once_in_test_and_once_in_val():
    vessels = vessel_rows()
    rows = cv.make_cv_splits(vessels)
    big = {v["vessel_group"] for v in vessels if v["vessel_class"] not in SMALL}
    for r in range(cv.N_REPETITIONS):
        for s in ("test", "val"):
            count = defaultdict(set)
            for p in rows:
                if int(p["repetition"]) == r and p["split"] == s:
                    count[p["vessel_group"]].add(int(p["fold"]))
            assert all(len(count[g]) == 1 for g in big)


def test_synthetic_every_class_in_train_val_and_test_in_every_partition():
    rows = cv.make_cv_splits(vessel_rows())
    for part in cv.partitions(rows).values():
        present = defaultdict(set)
        for p in part:
            present[(p["vessel_class"], p["split"])].add(p["vessel_group"])
        for c in ("Tug", "Four", "A", "B"):
            for s in cv.SPLITS:
                assert present[(c, s)], (c, s)
        assert [len(present[("Tug", s)]) for s in cv.SPLITS] == [1, 1, 1]
        assert [len(present[("Four", s)]) for s in cv.SPLITS] == [2, 1, 1]


def test_synthetic_small_classes_balanced_in_test_and_val():
    counts = cv.role_counts(cv.make_cv_splits(vessel_rows()))
    for cls, names in SMALL.items():
        low = 25 // len(names)
        for s in ("test", "val"):
            got = [counts.get((g, s), 0) for g in names]
            assert sum(got) == 25 and min(got) >= low and max(got) <= low + 1, (cls, s)


def test_big_class_folds_do_not_depend_on_small_classes():
    with_small = cv.make_cv_splits(vessel_rows())
    without = cv.make_cv_splits(
        vessel_rows([x for x in SYNTHETIC if x[0].split("/")[0] not in SMALL])
    )
    assert [p for p in with_small if p["vessel_class"] not in SMALL] == without


def test_check_detects_a_vessel_leak():
    vessels = vessel_rows()
    rows = cv.make_cv_splits(vessels)
    # B/9.wav belongs to vessel B/W0, like B/0.wav: put it in another role in one partition.
    for p in rows:
        if p["repetition"] == "0" and p["fold"] == "0" and p["wav_path"] == "B/9.wav":
            p["split"] = "test" if p["split"] != "test" else "train"
    errors = cv.check_cv_splits(rows, vessels)
    assert any("vessel leak: B/W0" in e for e in errors)


def test_check_detects_a_missing_tug_in_train():
    vessels = vessel_rows()
    rows = cv.make_cv_splits(vessels)
    for p in rows:
        if (p["repetition"], p["fold"], p["vessel_class"], p["split"]) == (
            "1",
            "2",
            "Tug",
            "train",
        ):
            p["split"] = "val"
    errors = cv.check_cv_splits(rows, vessels)
    assert "repetition 1 fold 2: class Tug has no group in train" in errors
    assert (
        "repetition 1 fold 2: class Tug has 2 groups in val (expected exactly one)"
        in errors
    )


def test_check_detects_unbalanced_small_class():
    vessels = vessel_rows()
    rows = cv.make_cv_splits(vessels)
    # Put Tug/T1 in test in every partition (swapping roles with the test vessel).
    for part in cv.partitions(rows).values():
        test = next(
            p for p in part if p["vessel_class"] == "Tug" and p["split"] == "test"
        )
        t1 = next(p for p in part if p["vessel_group"] == "Tug/T1")
        test["split"], t1["split"] = t1["split"], "test"
    errors = cv.check_cv_splits(rows, vessels)
    assert any(e.startswith("class Tug: times in test per group") for e in errors)


def test_main_writes_lf_utf8_and_is_reproducible(tmp_path, capsys):
    vessels = write_vessels(tmp_path / "vessels.csv", SYNTHETIC)
    out1, out2 = tmp_path / "a.csv", tmp_path / "b.csv"
    assert cv.main(["--vessels", str(vessels), "--output", str(out1), "--summary"]) == 0
    assert cv.main(["--vessels", str(vessels), "--output", str(out2)]) == 0
    raw = out1.read_bytes()
    assert raw == out2.read_bytes()
    assert b"\r" not in raw
    assert (
        raw.splitlines()[0]
        == b"repetition,fold,wav_path,vessel_class,vessel_group,split"
    )
    assert len(raw.splitlines()) == 1 + 25 * len(SYNTHETIC)
    printed = capsys.readouterr().out
    assert "worst maxdev" in printed and "repetition 4 fold 4" in printed
    assert "Tug (fewer than 5 vessels): partitions in test / val / train" in printed


def test_main_fails_and_writes_nothing_with_too_few_groups(tmp_path, capsys):
    vessels = write_vessels(
        tmp_path / "vessels.csv",
        [*SYNTHETIC[3:], ("Tug/1.wav", "Tug/T1", 100), ("Tug/2.wav", "Tug/T2", 90)],
    )
    out = tmp_path / "cv.csv"
    assert cv.main(["--vessels", str(vessels), "--output", str(out)]) == 1
    assert not out.exists()
    assert "class Tug" in capsys.readouterr().err


def test_main_writes_nothing_if_leak_check_fails(tmp_path, monkeypatch, capsys):
    vessels = write_vessels(tmp_path / "vessels.csv", SYNTHETIC)
    monkeypatch.setattr(cv.check_leaks, "check_split", lambda s, v: ["vessel leak: X"])
    out = tmp_path / "cv.csv"
    assert cv.main(["--vessels", str(vessels), "--output", str(out)]) == 1
    assert not out.exists()
    assert "vessel leak: X" in capsys.readouterr().err
