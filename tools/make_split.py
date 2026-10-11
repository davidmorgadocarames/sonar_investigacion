"""Make the train/val/test split by vessel (task F1-T06c, D-007, D-012, D-013).

Reads only ``datasets/manifests/vessels.csv`` (no audio) and writes
``datasets/manifests/split.csv`` (D-012 format: one row per WAV, in the order of
``vessels.csv``, UTF-8, LF)::

    wav_path,vessel_class,vessel_group,split

Unit of assignment: the vessel group (``vessel_group``). A group always goes whole to one split,
so a vessel can never be in two splits. Each class is split on its own.

Objective (D-013): 60/20/20 (train/val/test) measured in seconds of audio *within each class*.
The duration of a group is the sum of ``wav_duration_s`` of its WAVs. For one class with total
duration T and per-split durations d_s, the cost is

    cost = sum over s in (train, val, test) of | d_s / T - target_s |

i.e. the L1 distance between the real and the target fractions (0 = exact, maximum 1.6). It is
computed exactly in integers: durations in microseconds (the 6 decimals of ``wav_duration_s``)
and ``5 * T * cost = |5 d_train - 3 T| + |5 d_val - T| + |5 d_test - T|``.

Constraint: every split gets at least one group of every class (so Tug, with 3 groups, is 1/1/1).

Search: exact branch and bound, no randomness (so no seed is needed). Groups are visited in a
fixed order, duration descending and then name ascending, and each one is tried in train, val
and test, in that order. A branch is cut when its lower bound is not better than the best cost
found so far. Lower bound: the excess of a split over its target can only grow, and since the
fractions add up to 1 the final cost is at least twice the current total excess. A branch is also
cut when fewer groups remain than splits are still empty.

Ties: because the search visits assignments in lexicographic order (group order above, then
train < val < test) and only a strictly lower cost replaces the best one, the result is the
lexicographically first assignment among those with minimal cost. The result is therefore unique
and reproducible: running the script again writes the same bytes.

Before writing, the split is validated with ``tools/check_leaks.py`` (``check_split``) and the
"at least one group per split and class" constraint; if anything fails, nothing is written and
the script exits with code 1.

Usage (inside Docker, from /app):
    docker compose run --rm --entrypoint python pipeline tools/make_split.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import sys
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VESSELS = ROOT / "datasets" / "manifests" / "vessels.csv"
DEFAULT_OUTPUT = ROOT / "datasets" / "manifests" / "split.csv"

SPLITS = ("train", "val", "test")
# D-013 targets as integer fifths: 3/5, 1/5, 1/5 (exact integer arithmetic).
TARGET_FIFTHS = (3, 1, 1)
SPLIT_FIELDS = ["wav_path", "vessel_class", "vessel_group", "split"]
MICROSECONDS = Decimal(10**6)


def _load_check_leaks():
    path = Path(__file__).resolve().parent / "check_leaks.py"
    spec = importlib.util.spec_from_file_location("check_leaks", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


check_leaks = _load_check_leaks()


class SplitError(ValueError):
    pass


@dataclass(frozen=True)
class Group:
    name: str
    duration_us: int
    n_wav: int


def duration_us(value: str) -> int:
    """Exact duration in integer microseconds from a ``wav_duration_s`` string."""
    us = Decimal(value) * MICROSECONDS
    if us != us.to_integral_value() or us < 0:
        raise SplitError(f"duration {value!r} is not a non-negative multiple of 1 us")
    return int(us)


def scaled_cost(sums: tuple[int, ...], total: int) -> int:
    """5 * total * cost, where cost = sum |d_s / total - target_s| (exact integer)."""
    return sum(abs(5 * d - f * total) for d, f in zip(sums, TARGET_FIFTHS))


def search_order(groups: list[Group]) -> list[Group]:
    """Fixed visiting order: duration descending, then name ascending."""
    return sorted(groups, key=lambda g: (-g.duration_us, g.name))


def assign_class(groups: list[Group]) -> dict[str, str]:
    """Optimal assignment of the groups of one class to the splits (see module docstring).

    Returns {group name: split}. Raises SplitError if the class has fewer than 3 groups.
    """
    names = [g.name for g in groups]
    if len(set(names)) != len(names):
        raise SplitError(f"duplicate group names: {sorted(names)}")
    if len(groups) < len(SPLITS):
        raise SplitError(
            f"{len(groups)} group(s) {sorted(names)}: need at least {len(SPLITS)}"
        )
    order = search_order(groups)
    total = sum(g.duration_us for g in order)
    targets = [f * total for f in TARGET_FIFTHS]  # in fifths of microseconds
    n = len(order)

    best_cost: int | None = None
    best: list[int] = []
    current: list[int] = []
    sums = [0, 0, 0]
    counts = [0, 0, 0]

    def visit(i: int) -> None:
        nonlocal best_cost, best
        empty = sum(1 for c in counts if c == 0)
        if empty > n - i:
            return
        excess = sum(max(0, 5 * s - t) for s, t in zip(sums, targets))
        if best_cost is not None and 2 * excess >= best_cost:
            return
        if i == n:
            best_cost = scaled_cost(tuple(sums), total)
            best = list(current)
            return
        for k in range(len(SPLITS)):
            sums[k] += order[i].duration_us
            counts[k] += 1
            current.append(k)
            visit(i + 1)
            current.pop()
            counts[k] -= 1
            sums[k] -= order[i].duration_us

    visit(0)
    return {g.name: SPLITS[k] for g, k in zip(order, best)}


def class_groups(vessel_rows: list[dict[str, str]]) -> dict[str, list[Group]]:
    """Groups per class, with their summed duration and number of WAVs."""
    duration: dict[tuple[str, str], int] = defaultdict(int)
    n_wav: dict[tuple[str, str], int] = defaultdict(int)
    for r in vessel_rows:
        key = (r["vessel_class"], r["vessel_group"])
        duration[key] += duration_us(r["wav_duration_s"])
        n_wav[key] += 1
    classes: dict[str, list[Group]] = defaultdict(list)
    for (cls, group), us in sorted(duration.items()):
        classes[cls].append(Group(group, us, n_wav[(cls, group)]))
    return dict(classes)


def make_split(vessel_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Split rows in the order of vessels.csv (D-012)."""
    assignment: dict[str, str] = {}
    for cls, groups in class_groups(vessel_rows).items():
        try:
            assignment.update(assign_class(groups))
        except SplitError as exc:
            raise SplitError(f"class {cls}: {exc}") from exc
    return [
        {
            "wav_path": r["wav_path"],
            "vessel_class": r["vessel_class"],
            "vessel_group": r["vessel_group"],
            "split": assignment[r["vessel_group"]],
        }
        for r in vessel_rows
    ]


def coverage_errors(split_rows: list[dict[str, str]]) -> list[str]:
    """Classes without at least one group in some split (D-013)."""
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in split_rows:
        groups[(r["vessel_class"], r["split"])].add(r["vessel_group"])
    classes = sorted({r["vessel_class"] for r in split_rows})
    return [
        f"class {c} has no group in {s}"
        for c in classes
        for s in SPLITS
        if not groups[(c, s)]
    ]


def render(split_rows: list[dict[str, str]]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=SPLIT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(split_rows)
    return out.getvalue()


def report(split_rows: list[dict[str, str]], vessel_rows: list[dict[str, str]]) -> str:
    """Minutes, groups and WAVs per class x split, real fraction against the target."""
    seconds = {r["wav_path"]: float(r["wav_duration_s"]) for r in vessel_rows}
    header = (
        f"{'class':<14}{'split':<7}{'groups':>7}{'WAV':>5}{'min':>8}"
        f"{'frac':>8}{'target':>8}{'diff':>8}"
    )
    lines = [header, "-" * len(header)]
    classes = sorted({r["vessel_class"] for r in split_rows})
    for cls in [*classes, "ALL"]:
        rows = [r for r in split_rows if cls in ("ALL", r["vessel_class"])]
        total = sum(seconds[r["wav_path"]] for r in rows)
        cost = 0.0
        for split, fifths in zip(SPLITS, TARGET_FIFTHS):
            part = [r for r in rows if r["split"] == split]
            s = sum(seconds[r["wav_path"]] for r in part)
            frac, target = s / total, fifths / 5
            cost += abs(frac - target)
            lines.append(
                f"{cls:<14}{split:<7}{len({r['vessel_group'] for r in part}):>7}"
                f"{len(part):>5}{s / 60:>8.2f}{frac:>8.3f}{target:>8.3f}"
                f"{frac - target:>+8.3f}"
            )
        lines.append(
            f"{cls:<14}{'total':<7}{len({r['vessel_group'] for r in rows}):>7}"
            f"{len(rows):>5}{total / 60:>8.2f}   cost = sum|frac - target| = {cost:.4f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vessels", type=Path, default=DEFAULT_VESSELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        vessel_rows = check_leaks.read_csv(args.vessels, check_leaks.VESSEL_FIELDS)
        split_rows = make_split(vessel_rows)
    except (check_leaks.ManifestError, SplitError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    errors = check_leaks.check_split(split_rows, vessel_rows) + coverage_errors(
        split_rows
    )
    if errors:
        print(f"FAIL: {len(errors)} problem(s); nothing written", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        f.write(render(split_rows))
    print(report(split_rows, vessel_rows))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
