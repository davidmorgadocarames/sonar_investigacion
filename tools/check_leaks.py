"""Check a train/val/test split manifest for data leaks (task F1-T06b, D-007).

The split manifest (``datasets/manifests/split.csv``, generated in F1-T06c) has one row per WAV
with at least these columns::

    wav_path,vessel_class,vessel_group,split

``wav_path`` is relative to ``raw/DeepShip/``, as in ``vessels.csv``. It is checked against the
vessel table (``datasets/manifests/vessels.csv``), which is the reference for the class, the
group and the SHA-256 of every WAV. The check fails if:

- a ``split`` value is not one of train/val/test;
- a WAV appears more than once in the split, or two rows have the same SHA-256 (the same
  recording under two paths);
- a WAV of the split is not in the vessel table, or a WAV of the vessel table is missing;
- the class or group of a row does not match the vessel table;
- a vessel group (as declared in the split or as given by the vessel table) is in two splits.

With ``--similar-names`` it prints instead the vessel groups with WAVs whose names look alike
(DT-003), for a human to review. The criteria are explicit in ``similar_names``.

Usage (inside Docker, from /app):
    docker compose run --rm --entrypoint python pipeline tools/check_leaks.py [split.csv]
    docker compose run --rm --entrypoint python pipeline tools/check_leaks.py --similar-names
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = ROOT / "datasets" / "manifests" / "split.csv"
DEFAULT_VESSELS = ROOT / "datasets" / "manifests" / "vessels.csv"

SPLITS = ("train", "val", "test")
SPLIT_FIELDS = ("wav_path", "vessel_class", "vessel_group", "split")
VESSEL_FIELDS = ("wav_path", "vessel_class", "vessel_group", "sha256", "wav_duration_s")

# DT-003: names closer than this (Levenshtein, on the normalised name) are listed.
MAX_EDIT_DISTANCE = 3


class ManifestError(ValueError):
    pass


def read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ManifestError(f"{path}: file not found")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError(f"{path}: missing columns {missing}")
        return list(reader)


# --------------------------------------------------------------------------- leaks


def _spread(rows: list[dict[str, str]], key) -> dict[str, set[str]]:
    """Map each key value to the set of splits it appears in."""
    spread: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        spread[key(r)].add(r["split"])
    return spread


def _fmt(splits: set[str]) -> str:
    ordered = [s for s in SPLITS if s in splits] + sorted(splits - set(SPLITS))
    return "/".join(ordered)


def check_split(
    split_rows: list[dict[str, str]], vessel_rows: list[dict[str, str]]
) -> list[str]:
    """Return the list of problems found; an empty list means the split has no leaks."""
    errors: list[str] = []
    vessels = {r["wav_path"]: r for r in vessel_rows}

    # The reference table must itself be one row per recording.
    for path, n in sorted(Counter(r["wav_path"] for r in vessel_rows).items()):
        if n > 1:
            errors.append(f"vessels.csv: {path} appears {n} times")
    sha_paths: dict[str, list[str]] = defaultdict(list)
    for r in vessel_rows:
        sha_paths[r["sha256"]].append(r["wav_path"])
    for sha, paths in sorted(sha_paths.items()):
        if len(paths) > 1:
            errors.append(
                f"vessels.csv: same content (sha256 {sha[:12]}...) in {sorted(paths)}"
            )

    for i, r in enumerate(split_rows, start=2):  # line 1 is the header
        if r["split"] not in SPLITS:
            errors.append(
                f"line {i}: {r['wav_path']}: invalid split {r['split']!r} "
                f"(expected one of {', '.join(SPLITS)})"
            )

    # Recording leaks: the same path twice, or the same content under two paths.
    for path, n in sorted(Counter(r["wav_path"] for r in split_rows).items()):
        if n > 1:
            splits = {r["split"] for r in split_rows if r["wav_path"] == path}
            errors.append(
                f"recording leak: {path} appears {n} times (splits {_fmt(splits)})"
            )
    by_sha: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in split_rows:
        if r["wav_path"] in vessels:
            by_sha[vessels[r["wav_path"]]["sha256"]].append(r)
    for sha, rows in sorted(by_sha.items()):
        paths = sorted({r["wav_path"] for r in rows})
        if len(paths) > 1:
            errors.append(
                f"recording leak: same content (sha256 {sha[:12]}...) under {paths} "
                f"(splits {_fmt({r['split'] for r in rows})})"
            )

    # Coverage and consistency with the vessel table.
    split_paths = {r["wav_path"] for r in split_rows}
    for path in sorted(split_paths - vessels.keys()):
        errors.append(f"unknown WAV: {path} is not in vessels.csv")
    for path in sorted(vessels.keys() - split_paths):
        errors.append(f"missing WAV: {path} from vessels.csv is not in the split")
    for r in split_rows:
        ref = vessels.get(r["wav_path"])
        if ref is None:
            continue
        for column in ("vessel_class", "vessel_group"):
            if r[column] != ref[column]:
                errors.append(
                    f"{column} mismatch: {r['wav_path']} has {r[column]!r} in the split "
                    f"but {ref[column]!r} in vessels.csv"
                )

    # Vessel leaks, both on the declared group and on the reference group.
    declared = _spread(split_rows, lambda r: r["vessel_group"])
    reference = _spread(
        [r for r in split_rows if r["wav_path"] in vessels],
        lambda r: vessels[r["wav_path"]]["vessel_group"],
    )
    leaked = {g: s for g, s in declared.items() if len(s) > 1}
    for g, s in reference.items():
        if len(s) > 1:
            leaked.setdefault(g, set()).update(s)
    for group, splits in sorted(leaked.items()):
        errors.append(f"vessel leak: {group} is in {_fmt(splits)}")
    return errors


def summarize(
    split_rows: list[dict[str, str]], vessel_rows: list[dict[str, str]]
) -> str:
    """WAV, groups and hours per split and per split x class."""
    duration = {r["wav_path"]: float(r["wav_duration_s"]) for r in vessel_rows}

    def line(label: str, rows: list[dict[str, str]]) -> str:
        hours = sum(duration[r["wav_path"]] for r in rows) / 3600
        groups = len({r["vessel_group"] for r in rows})
        return f"{label:<22} {len(rows):>4} WAV {groups:>4} groups {hours:>7.3f} h"

    classes = sorted({r["vessel_class"] for r in split_rows})
    out = [line("all", split_rows)]
    for split in SPLITS:
        rows = [r for r in split_rows if r["split"] == split]
        out.append(line(split, rows))
        for c in classes:
            out.append(line(f"  {c}", [r for r in rows if r["vessel_class"] == c]))
    return "\n".join(out)


# --------------------------------------------------------------------------- DT-003


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def name_criteria(a: str, b: str) -> list[str]:
    """Explicit criteria for two normalised vessel names to look alike (DT-003)."""
    criteria = []
    if a == b:
        criteria.append("same_name")
    elif a.startswith(b) or b.startswith(a):
        criteria.append("prefix")
    shared = sorted(set(a.split()) & set(b.split()))
    if shared:
        criteria.append("shared_word:" + " ".join(shared))
    if a != b and edit_distance(a, b) <= MAX_EDIT_DISTANCE:
        criteria.append(f"edit_distance<={MAX_EDIT_DISTANCE}")
    return criteria


def similar_names(vessel_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Pairs of vessel groups with WAVs whose names look alike, within and across classes."""
    wavs = Counter(r["vessel_group"] for r in vessel_rows)
    groups = sorted(wavs)
    pairs = []
    for ga, gb in combinations(groups, 2):
        class_a, name_a = ga.split("/", 1)
        class_b, name_b = gb.split("/", 1)
        criteria = name_criteria(name_a, name_b)
        if criteria:
            pairs.append(
                {
                    "scope": "same_class" if class_a == class_b else "cross_class",
                    "group_a": ga,
                    "wav_a": str(wavs[ga]),
                    "group_b": gb,
                    "wav_b": str(wavs[gb]),
                    "criteria": "+".join(criteria),
                }
            )
    pairs.sort(key=lambda p: (p["scope"] != "same_class", p["group_a"], p["group_b"]))
    return pairs


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("split", nargs="?", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--vessels", type=Path, default=DEFAULT_VESSELS)
    parser.add_argument(
        "--similar-names",
        action="store_true",
        help="list vessel groups with similar names (DT-003) instead of checking a split",
    )
    args = parser.parse_args(argv)
    try:
        vessel_rows = read_csv(args.vessels, VESSEL_FIELDS)
        if args.similar_names:
            pairs = similar_names(vessel_rows)
            writer = csv.DictWriter(
                sys.stdout,
                fieldnames=list(pairs[0]) if pairs else ["scope"],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(pairs)
            return 0
        if not args.split.is_file():
            raise ManifestError(
                f"{args.split}: file not found (the split is generated in F1-T06c)"
            )
        split_rows = read_csv(args.split, SPLIT_FIELDS)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    errors = check_split(split_rows, vessel_rows)
    if errors:
        print(f"FAIL: {len(errors)} problem(s) in {args.split}", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    print(f"OK: no leaks in {args.split}")
    print(summarize(split_rows, vessel_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
