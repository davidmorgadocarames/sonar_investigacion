"""Make the 25 partitions of the 5 x 5 repeated cross-validation by vessel (F5-T02, D-024).

Reads only ``datasets/manifests/vessels.csv`` (no audio) and writes
``datasets/manifests/cv_splits.csv`` (UTF-8, LF), one row per (repetition, fold, WAV)::

    repetition,fold,wav_path,vessel_class,vessel_group,split

Rows are ordered by repetition, then fold, then the order of ``vessels.csv``.

Unit of assignment: the vessel group (``vessel_group``). A group always goes whole to one fold,
so a vessel can never be in two roles of the same partition. Each class is split on its own.

Folds of one repetition r (r = 0..4, ``numpy.random.default_rng(r)``; classes are processed in
alphabetical order with the same generator):

* Class with n >= 5 groups. Goal: the minutes of the class are balanced across the 5 folds,
  every fold has at least one group of the class, and the repetitions group the vessels
  differently. Durations are exact integers (microseconds); T = class total, d_f = fold f::

      cost          = sum over folds of (5 d_f - T)^2      (what the local search lowers)
      max_deviation = max over folds of |5 d_f - T|         (what the acceptance checks)

  0. Tolerance (no randomness): the reference assignment is the greedy one over the groups by
     duration descending (then name), each to the fold with the least duration so far (ties:
     lowest index), improved by the full local search below. The accepted max_deviation is the
     reference's plus T / 10, i.e. each fold may be up to 10 % of T/5 further from T/5 than in
     the reference.
  1. Random start: the groups (sorted by name) are permuted with the generator and assigned
     greedily in that order as above. The first five fill the five folds.
  2. Local search: among all moves of one group to another fold (only if its fold keeps at
     least one group) and all swaps of two groups in different folds, apply the one that lowers
     the cost the most (ties: first in the enumeration order, groups by name). Stop as soon as
     max_deviation is within the tolerance, or at a local optimum. If the result is within the
     tolerance it is accepted; otherwise a new random start is drawn (at most 1000 attempts).

  Why not "shuffle, sort by duration, greedy" or "local search to the optimum": the durations
  of the groups are nearly all distinct, so both give the same grouping of the vessels in every
  repetition (only the fold labels change), which would defeat the repetition. The tolerance
  trades a little balance for different groupings; the summary reports both.
* Class with 3 <= n < 5 groups (today only Tug, with 3; approved change to D-024): 5 folds
  cannot give every partition a vessel of the class in test and in validation, so these classes
  do not use folds. In every partition exactly 1 vessel is in test, 1 in validation and the rest
  (1 or 2) in training. The (test, val) pairs of the 25 partitions, taken in order (repetition,
  fold), form one rotation built in *blocks* of n partitions. A block has a shift s in 1..n-1
  (vessels sorted by name, indices modulo n): its n partitions take every vessel x once as test,
  each with val = x + s. So in a block every vessel is in test exactly once and in validation
  exactly once, and in a block cut short (the last one) the test vessels are all different and
  so are the val vessels. Hence over the 25 partitions each vessel is in test floor(25/n) or
  ceil(25/n) times (with 3 vessels: 9/8/8) and the same for validation: the +-1 balance holds by
  construction, and test != val always since s != 0.
  Randomness, with the generator of the repetition the partition belongs to: the shift of a new
  block is drawn among the shifts different from the previous block's, and the test vessel of
  each partition is drawn among those not yet used in the block, preferring the ones whose
  (test, val) pair has not been used yet in the same repetition (so the 5 partitions of a
  repetition repeat no pair whenever the block structure allows it). The block state carries
  over from one repetition to the next (that is what keeps the global balance). Big classes draw
  first in each repetition, so their folds do not depend on the small classes.
* Class with fewer than 3 groups: error (it could not be in train, val and test at once).

Partition (r, k): for classes with n >= 5 groups, test = fold k, validation = fold (k + 1) % 5,
training = the other three folds, so each of their groups is in test exactly once and in
validation exactly once per repetition. Together with the rule above, every class has at least
one vessel in train, val and test in every partition. With a dataset where every class has at
least 5 vessels, everything would follow the 5-fold rule (60/20/20 in vessels).

Before writing, every partition is validated with ``check_split`` of ``tools/check_leaks.py``
plus the coverage checks above; if anything fails, nothing is written and the exit code is 1.

Usage (inside Docker, from /app):
    docker compose run --rm --entrypoint python pipeline tools/make_cv_splits.py [--summary]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VESSELS = ROOT / "datasets" / "manifests" / "vessels.csv"
DEFAULT_OUTPUT = ROOT / "datasets" / "manifests" / "cv_splits.csv"

N_REPETITIONS = 5
N_FOLDS = 5
MIN_GROUPS = 3
# Balance slack over the reference optimum, in tenths of 1/5 of the class (see class_tolerance).
SLACK_TENTHS = 1
MAX_ATTEMPTS = 1000
SPLITS = ("train", "val", "test")
CV_FIELDS = ["repetition", "fold", "wav_path", "vessel_class", "vessel_group", "split"]


def _load_make_split():
    path = Path(__file__).resolve().parent / "make_split.py"
    spec = importlib.util.spec_from_file_location("make_split", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


make_split = _load_make_split()
check_leaks = make_split.check_leaks
Group = make_split.Group


class CVSplitError(ValueError):
    pass


def fold_cost(sums: list[int], total: int) -> int:
    """sum_f (5 d_f - T)^2: exact integer imbalance of one class across the folds."""
    return sum((N_FOLDS * d - total) ** 2 for d in sums)


def max_deviation(sums: list[int], total: int) -> int:
    """max_f |5 d_f - T|: 5 times the largest gap between a fold and 1/5 of the class."""
    return max(abs(N_FOLDS * d - total) for d in sums)


def _sums(durations: list[int], folds: list[int]) -> list[int]:
    sums = [0] * N_FOLDS
    for d, f in zip(durations, folds):
        sums[f] += d
    return sums


def _local_search(
    durations: list[int], folds: list[int], total: int, tolerance: int | None = None
) -> list[int]:
    """Best-improvement moves and swaps on fold_cost (see module docstring).

    Stops at a local optimum or, if ``tolerance`` is given, as soon as
    ``max_deviation <= tolerance``.
    """
    folds = list(folds)
    n = len(durations)
    sums = _sums(durations, folds)
    counts = [folds.count(f) for f in range(N_FOLDS)]
    cost = fold_cost(sums, total)
    while tolerance is None or max_deviation(sums, total) > tolerance:
        best_cost, best_step = cost, None
        for i in range(n):
            a = folds[i]
            if counts[a] < 2:
                continue
            for b in range(N_FOLDS):
                if b == a:
                    continue
                trial = list(sums)
                trial[a] -= durations[i]
                trial[b] += durations[i]
                c = fold_cost(trial, total)
                if c < best_cost:
                    best_cost, best_step = c, ("move", i, b)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = folds[i], folds[j]
                if a == b or durations[i] == durations[j]:
                    continue
                trial = list(sums)
                delta = durations[j] - durations[i]
                trial[a] += delta
                trial[b] -= delta
                c = fold_cost(trial, total)
                if c < best_cost:
                    best_cost, best_step = c, ("swap", i, j)
        if best_step is None:
            break
        kind, i, x = best_step
        if kind == "move":
            a = folds[i]
            sums[a] -= durations[i]
            sums[x] += durations[i]
            counts[a] -= 1
            counts[x] += 1
            folds[i] = x
        else:
            a, b = folds[i], folds[x]
            delta = durations[x] - durations[i]
            sums[a] += delta
            sums[b] -= delta
            folds[i], folds[x] = b, a
        cost = best_cost
    return folds


def _greedy(durations: list[int], order) -> list[int]:
    """Each group, in ``order``, to the fold with the least duration so far (ties: lowest)."""
    folds = [0] * len(durations)
    sums = [0] * N_FOLDS
    for i in order:
        i = int(i)
        f = min(range(N_FOLDS), key=lambda k: (sums[k], k))
        folds[i] = f
        sums[f] += durations[i]
    return folds


def class_tolerance(durations: list[int]) -> int:
    """Accepted max_deviation for a class: reference optimum + SLACK_TENTHS/10 of T.

    The reference is the deterministic, seed-free optimum: greedy by duration descending
    (then name) followed by the full local search. In minutes this allows each fold to be
    up to 10 % of T/5 further from T/5 than the reference.
    """
    total = sum(durations)
    order = sorted(range(len(durations)), key=lambda i: (-durations[i], i))
    reference = _local_search(durations, _greedy(durations, order), total)
    return (
        max_deviation(_sums(durations, reference), total) + SLACK_TENTHS * total // 10
    )


def assign_folds(groups: list[Group], rng: np.random.Generator) -> dict[str, int]:
    """Fold (0..4) of every group of one class (see module docstring)."""
    order = sorted(groups, key=lambda g: g.name)
    names = [g.name for g in order]
    if len(set(names)) != len(names):
        raise CVSplitError(f"duplicate group names: {names}")
    if len(order) < MIN_GROUPS:
        raise CVSplitError(f"{len(order)} group(s) {names}: need at least {MIN_GROUPS}")
    if any(g.duration_us <= 0 for g in order):
        raise CVSplitError(f"groups with non-positive duration in {names}")
    if len(order) < N_FOLDS:
        raise CVSplitError(f"{len(order)} group(s): folds need at least {N_FOLDS}")

    durations = [g.duration_us for g in order]
    total = sum(durations)
    tolerance = class_tolerance(durations)
    for _ in range(MAX_ATTEMPTS):
        start = _greedy(durations, rng.permutation(len(order)))
        folds = _local_search(durations, start, total, tolerance)
        if max_deviation(_sums(durations, folds), total) <= tolerance:
            return {g.name: f for g, f in zip(order, folds)}
    raise CVSplitError(f"no balanced assignment in {MAX_ATTEMPTS} attempts")


class SmallClassRotation:
    """(test, val) vessels of a class with 3 <= n < 5 groups, partition by partition.

    See the module docstring: blocks of n partitions, each with a shift s != 0 and every vessel
    once as test (val = test + s). The state carries over across repetitions.
    """

    def __init__(self, groups: list[Group]):
        names = sorted(g.name for g in groups)
        if len(set(names)) != len(names):
            raise CVSplitError(f"duplicate group names: {names}")
        if not MIN_GROUPS <= len(names) < N_FOLDS:
            raise CVSplitError(
                f"{len(names)} group(s) {names}: rotation needs {MIN_GROUPS} to {N_FOLDS - 1}"
            )
        self.names = names
        self.shift = 0  # 0 = no block yet
        self.remaining: list[int] = []

    def next_pair(
        self, rng: np.random.Generator, used: set[tuple[int, int]]
    ) -> tuple[int, int]:
        """Indices (test, val) of the next partition; ``used`` = pairs of this repetition."""
        n = len(self.names)
        if not self.remaining:
            shifts = [s for s in range(1, n) if s != self.shift]
            self.shift = shifts[int(rng.integers(len(shifts)))]
            self.remaining = list(range(n))
        fresh = [x for x in self.remaining if (x, (x + self.shift) % n) not in used]
        pool = fresh or self.remaining
        x = pool[int(rng.integers(len(pool)))]
        self.remaining.remove(x)
        return x, (x + self.shift) % n


def validate_groups(cls: str, groups: list[Group]) -> None:
    names = sorted(g.name for g in groups)
    if len(set(names)) != len(names):
        raise CVSplitError(f"class {cls}: duplicate group names: {names}")
    if len(groups) < MIN_GROUPS:
        raise CVSplitError(
            f"class {cls}: {len(groups)} group(s) {names}: need at least {MIN_GROUPS}"
        )


def assign_roles(
    class_groups: dict[str, list[Group]],
) -> dict[tuple[int, int], dict[str, str]]:
    """{(repetition, fold): {vessel group: role}} for the 25 partitions."""
    for cls, groups in class_groups.items():
        validate_groups(cls, groups)
    big = sorted(c for c, g in class_groups.items() if len(g) >= N_FOLDS)
    small = sorted(c for c, g in class_groups.items() if len(g) < N_FOLDS)
    rotations = {c: SmallClassRotation(class_groups[c]) for c in small}
    roles: dict[tuple[int, int], dict[str, str]] = {}
    for r in range(N_REPETITIONS):
        rng = np.random.default_rng(r)
        folds: dict[str, int] = {}
        for cls in big:
            try:
                folds.update(assign_folds(class_groups[cls], rng))
            except CVSplitError as exc:
                raise CVSplitError(f"class {cls}: {exc}") from exc
        used: dict[str, set[tuple[int, int]]] = {c: set() for c in small}
        for k in range(N_FOLDS):
            part = {g: role(f, k) for g, f in folds.items()}
            for cls in small:
                rot = rotations[cls]
                t, v = rot.next_pair(rng, used[cls])
                used[cls].add((t, v))
                for i, name in enumerate(rot.names):
                    part[name] = "test" if i == t else "val" if i == v else "train"
            roles[(r, k)] = part
    return roles


def role(group_fold: int, k: int) -> str:
    """Role of a fold in partition k: test = k, val = k + 1 (circular), train otherwise."""
    if group_fold == k:
        return "test"
    if group_fold == (k + 1) % N_FOLDS:
        return "val"
    return "train"


def make_cv_splits(vessel_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """All rows, ordered by repetition, fold and the order of vessels.csv."""
    roles = assign_roles(make_split.class_groups(vessel_rows))
    rows: list[dict[str, str]] = []
    for r in range(N_REPETITIONS):
        for k in range(N_FOLDS):
            for v in vessel_rows:
                rows.append(
                    {
                        "repetition": str(r),
                        "fold": str(k),
                        "wav_path": v["wav_path"],
                        "vessel_class": v["vessel_class"],
                        "vessel_group": v["vessel_group"],
                        "split": roles[(r, k)][v["vessel_group"]],
                    }
                )
    return rows


def small_groups(vessel_rows: list[dict[str, str]]) -> dict[str, list[str]]:
    """{class: sorted groups} of the classes with fewer than 5 groups."""
    groups: dict[str, set[str]] = defaultdict(set)
    for v in vessel_rows:
        groups[v["vessel_class"]].add(v["vessel_group"])
    return {c: sorted(g) for c, g in sorted(groups.items()) if len(g) < N_FOLDS}


def role_counts(rows: list[dict[str, str]]) -> dict[tuple[str, str], int]:
    """{(vessel group, role): number of partitions} over all partitions."""
    seen = {(p["repetition"], p["fold"], p["vessel_group"], p["split"]) for p in rows}
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for _, _, g, s in seen:
        counts[(g, s)] += 1
    return dict(counts)


def partitions(
    rows: list[dict[str, str]],
) -> dict[tuple[int, int], list[dict[str, str]]]:
    """{(repetition, fold): rows of that partition}, in file order."""
    out: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[(int(row["repetition"]), int(row["fold"]))].append(row)
    return dict(out)


def check_cv_splits(
    rows: list[dict[str, str]], vessel_rows: list[dict[str, str]]
) -> list[str]:
    """Problems in the 25 partitions; an empty list means everything is fine."""
    errors: list[str] = []
    parts = partitions(rows)
    expected = {(r, k) for r in range(N_REPETITIONS) for k in range(N_FOLDS)}
    if set(parts) != expected:
        errors.append(f"partitions {sorted(parts)} != {sorted(expected)}")
    classes = sorted({v["vessel_class"] for v in vessel_rows})
    small = small_groups(vessel_rows)
    big_groups = {
        v["vessel_group"] for v in vessel_rows if v["vessel_class"] not in small
    }
    for (r, k), part in sorted(parts.items()):
        tag = f"repetition {r} fold {k}"
        errors += [f"{tag}: {e}" for e in check_leaks.check_split(part, vessel_rows)]
        present: dict[tuple[str, str], set[str]] = defaultdict(set)
        for p in part:
            present[(p["vessel_class"], p["split"])].add(p["vessel_group"])
        for c in classes:
            for s in SPLITS:
                if not present[(c, s)]:
                    errors.append(f"{tag}: class {c} has no group in {s}")
        for c in small:
            for s in ("test", "val"):
                if len(present[(c, s)]) != 1:
                    errors.append(
                        f"{tag}: class {c} has {len(present[(c, s)])} groups in {s}"
                        " (expected exactly one)"
                    )
    for r in range(N_REPETITIONS):
        for s in ("test", "val"):
            seen: dict[str, set[int]] = defaultdict(set)
            for k in range(N_FOLDS):
                for p in parts.get((r, k), []):
                    if p["split"] == s:
                        seen[p["vessel_group"]].add(k)
            for g in sorted(big_groups):
                if len(seen.get(g, ())) != 1:
                    errors.append(
                        f"repetition {r}: {g} is in {s} in folds {sorted(seen.get(g, ()))}"
                        " (expected exactly one)"
                    )
    counts = role_counts(rows)
    n_parts = len(parts)
    for c, groups in small.items():
        low = n_parts // len(groups)
        for s in ("test", "val"):
            got = [counts.get((g, s), 0) for g in groups]
            if min(got) < low or max(got) > low + 1:
                errors.append(f"class {c}: times in {s} per group {got} not within +-1")
    return errors


def render(rows: list[dict[str, str]]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def fold_minutes(
    rows: list[dict[str, str]], vessel_rows: list[dict[str, str]]
) -> dict[tuple[int, str], list[float]]:
    """{(repetition, class): minutes per fold}, classes with 5 or more groups only."""
    seconds = {v["wav_path"]: float(v["wav_duration_s"]) for v in vessel_rows}
    small = small_groups(vessel_rows)
    out: dict[tuple[int, str], list[float]] = defaultdict(lambda: [0.0] * N_FOLDS)
    for p in rows:
        if p["split"] == "test" and p["vessel_class"] not in small:
            out[(int(p["repetition"]), p["vessel_class"])][int(p["fold"])] += (
                seconds[p["wav_path"]] / 60
            )
    return dict(out)


def class_partitions(rows: list[dict[str, str]]) -> dict[tuple[int, str], frozenset]:
    """{(repetition, class): the folds of the class as a set of sets of groups (no labels).

    Only meaningful for classes with 5 or more groups (the others do not use folds).
    """
    members: dict[tuple[int, str, int], set[str]] = defaultdict(set)
    for p in rows:
        if p["split"] == "test":
            members[(int(p["repetition"]), p["vessel_class"], int(p["fold"]))].add(
                p["vessel_group"]
            )
    out: dict[tuple[int, str], set] = defaultdict(set)
    for (r, cls, _), groups in members.items():
        out[(r, cls)].add(frozenset(groups))
    return {key: frozenset(value) for key, value in out.items()}


def balance_report(
    rows: list[dict[str, str]], vessel_rows: list[dict[str, str]]
) -> str:
    """Minutes per class and fold in every repetition, with the max deviation from 1/5."""
    header = f"{'rep':<4}{'class':<14}" + "".join(
        f"{'f' + str(k):>7}" for k in range(N_FOLDS)
    )
    header += f"{'total':>8}{'maxdev':>8}"
    lines = [header, "-" * len(header)]
    worst: dict[str, float] = defaultdict(float)
    for (r, cls), mins in sorted(fold_minutes(rows, vessel_rows).items()):
        total = sum(mins)
        dev = max(abs(m - total / N_FOLDS) for m in mins) / (total / N_FOLDS)
        worst[cls] = max(worst[cls], dev)
        lines.append(
            f"{r:<4}{cls:<14}"
            + "".join(f"{m:>7.2f}" for m in mins)
            + f"{total:>8.2f}{dev:>+8.1%}"
        )
    lines.append("maxdev = max over folds of |minutes - total/5| / (total/5)")
    for cls, dev in sorted(worst.items()):
        lines.append(f"worst maxdev {cls:<14}{dev:.1%}")
    parts = class_partitions(rows)
    for cls in sorted(worst):
        distinct = len({parts[(r, cls)] for r in range(N_REPETITIONS)})
        lines.append(
            f"distinct groupings (ignoring fold labels) {cls:<14}{distinct}/{N_REPETITIONS}"
        )
    counts = role_counts(rows)
    for cls, groups in small_groups(vessel_rows).items():
        lines.append(
            f"{cls} (fewer than {N_FOLDS} vessels): partitions in test / val / train"
        )
        for g in groups:
            lines.append(
                f"  {g:<30}"
                + " / ".join(
                    str(counts.get((g, s), 0)) for s in ("test", "val", "train")
                )
            )
    return "\n".join(lines)


def partition_report(
    rows: list[dict[str, str]], vessel_rows: list[dict[str, str]]
) -> str:
    """Groups, WAVs and minutes per class and role in each of the 25 partitions."""
    seconds = {v["wav_path"]: float(v["wav_duration_s"]) for v in vessel_rows}
    classes = sorted({v["vessel_class"] for v in vessel_rows})
    lines = []
    for (r, k), part in sorted(partitions(rows).items()):
        lines.append(f"repetition {r} fold {k}")
        for cls in classes:
            cells = []
            for s in SPLITS:
                sel = [p for p in part if p["vessel_class"] == cls and p["split"] == s]
                n_groups = len({p["vessel_group"] for p in sel})
                minutes = sum(seconds[p["wav_path"]] for p in sel) / 60
                cells.append(
                    f"{s} {n_groups:>2} vessels {len(sel):>2} WAV {minutes:>6.2f} min"
                )
            lines.append(f"  {cls:<14}" + " | ".join(cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vessels", type=Path, default=DEFAULT_VESSELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="also print vessels, WAVs and minutes per class and role of every partition",
    )
    args = parser.parse_args(argv)
    try:
        vessel_rows = check_leaks.read_csv(args.vessels, check_leaks.VESSEL_FIELDS)
        rows = make_cv_splits(vessel_rows)
    except (check_leaks.ManifestError, make_split.SplitError, CVSplitError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    errors = check_cv_splits(rows, vessel_rows)
    if errors:
        print(f"FAIL: {len(errors)} problem(s); nothing written", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        f.write(render(rows))
    print(balance_report(rows, vessel_rows))
    if args.summary:
        print(partition_report(rows, vessel_rows))
    print(f"Wrote {args.output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
