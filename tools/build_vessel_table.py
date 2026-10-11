"""Build the WAV -> vessel table for the DeepShip subset (task F1-T06a, D-007).

Each class folder holds numbered WAV files and a ``*-metafile``: a header-less CSV with one row
per recording of the full dataset::

    id, AIS type code, vessel name, YYYYMMDD, hhmmss, duration_s, field7,

The WAV number is a recording number, but it does not always equal the metafile id (DT-002):
some ids are skipped or repeated, and the Tanker metafile restarts its numbering in three
blocks. The table is therefore built by an explainable alignment, without hand-written ranges:

1. Parse the metafile. A new *block* starts whenever the id goes down (the numbering restarts).
2. For each block, align the class WAVs (sorted by number) with the block rows (file order)
   keeping both orders: if WAV a < WAV b, then row(a) comes before row(b). Both lists are
   chronological, so a crossing assignment would mean time running backwards.
   A WAV may be paired with a row only if both durations agree within DURATION_TOLERANCE_S.
   The alignment maximises, in this order of priority:
     a. the number of WAVs paired with a row whose duration matches;
     b. how many of those pairs have id == WAV number;
     c. closeness of the id to the WAV number (minimal total |id - number|).
3. The class uses the block with the best alignment (the WAVs of a class come from one block).
4. A WAV left unpaired keeps the row whose id equals its number, marked ``id_unverified``.

The script fails instead of guessing (D-010, DT-006) if two blocks align equally well, if the
optimal alignment is not unique, if an unpaired WAV's id is missing or repeated in the block, if
a row would be used by two WAVs, or if a metafile line does not have the expected shape.

Vessel groups (D-007): class + normalised name (strip, upper case, spaces collapsed). Every name
starting with "QUEEN OF" (a truncated name) is one group, and BOWEN QUEEN joins it as a precaution
(D-011, DT-003). A group never spans two classes.

Usage (inside Docker, from /app):
    docker compose run --rm --entrypoint python pipeline tools/build_vessel_table.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "datasets" / "raw" / "DeepShip"
DEFAULT_OUTPUT = ROOT / "datasets" / "manifests" / "vessels.csv"

# Integer metafile durations against n_samples / sample_rate.
DURATION_TOLERANCE_S = 1.0

# D-007: truncated names merged into one group per class. A merge can never cause a leak.
MERGED_NAME_PREFIXES = ("QUEEN OF",)
# D-011 (DT-003): names merged into another group as a precaution, same class only.
MERGED_NAMES = {"BOWEN QUEEN": "QUEEN OF"}

SOURCE_ID = "id"
SOURCE_DURATION = "duration"
SOURCE_ID_UNVERIFIED = "id_unverified"

FIELDS = [
    "wav_path",
    "vessel_class",
    "wav_number",
    "sha256",
    "sample_rate",
    "n_samples",
    "wav_duration_s",
    "metafile",
    "metafile_line",
    "metafile_block",
    "metafile_id",
    "ais_code",
    "vessel_name_raw",
    "vessel_group",
    "date",
    "time",
    "metafile_duration_s",
    "field7",
    "match_source",
    "duration_match",
    "duration_diff_s",
]


class MetafileError(ValueError):
    pass


class AlignmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class MetaRow:
    line: int  # 1-based physical line in the metafile
    block: int  # 1-based numbering block
    id: int
    ais_code: str
    name_raw: str
    date: str
    time: str
    duration_s: int
    field7: str


@dataclass(frozen=True)
class Wav:
    path: Path
    number: int
    sample_rate: int
    n_samples: int

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate


# --------------------------------------------------------------------------- parsing


def parse_metafile(path: Path) -> list[MetaRow]:
    """Parse a DeepShip metafile. Blank lines are skipped; a decreasing id opens a new block."""
    rows: list[MetaRow] = []
    block = 1
    previous_id: int | None = None
    text = path.read_bytes().decode("utf-8")
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split(",")
        # 7 fields, usually followed by a trailing comma (and sometimes trailing whitespace).
        if len(fields) < 7 or any(f.strip() for f in fields[7:]):
            raise MetafileError(f"{path}:{line_no}: unexpected field count: {line!r}")
        try:
            row_id = int(fields[0].strip())
            duration = int(fields[5].strip())
        except ValueError as exc:
            raise MetafileError(
                f"{path}:{line_no}: bad id or duration: {line!r}"
            ) from exc
        if previous_id is not None and row_id < previous_id:
            block += 1
        previous_id = row_id
        rows.append(
            MetaRow(
                line=line_no,
                block=block,
                id=row_id,
                ais_code=fields[1].strip(),
                name_raw=fields[2],
                date=fields[3].strip(),
                time=fields[4].strip(),
                duration_s=duration,
                field7=fields[6],
            )
        )
    if not rows:
        raise MetafileError(f"{path}: no rows")
    return rows


def read_wav_header(path: Path) -> Wav:
    sample_rate, samples = wavfile.read(path, mmap=True)
    n_samples = int(samples.shape[0])
    del samples
    return Wav(
        path=path,
        number=int(path.stem),
        sample_rate=int(sample_rate),
        n_samples=n_samples,
    )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().upper())


def vessel_group(vessel_class: str, name_raw: str) -> str:
    name = normalize_name(name_raw)
    for prefix in MERGED_NAME_PREFIXES:
        if name.startswith(prefix):
            name = prefix
    name = MERGED_NAMES.get(name, name)
    return f"{vessel_class}/{name}"


# --------------------------------------------------------------------------- alignment


def durations_match(wav: Wav, row: MetaRow) -> bool:
    return abs(wav.duration_s - row.duration_s) <= DURATION_TOLERANCE_S


def pair_score(wav: Wav, row: MetaRow) -> tuple[int, int, int]:
    return (1, int(row.id == wav.number), -abs(row.id - wav.number))


def _add(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def align(
    wavs: list[Wav],
    rows: list[MetaRow],
    banned: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[tuple[int, int, int], list[tuple[int, int]]]:
    """Order-preserving alignment of wavs and rows that maximises the score (see module doc).

    Returns the score and the list of (wav index, row index) pairs. ``banned`` pairs are not
    allowed; it is used to test whether the optimum is unique.
    """
    m, k = len(wavs), len(rows)
    zero = (0, 0, 0)
    best = [[zero] * (k + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(k - 1, -1, -1):
            options = [best[i + 1][j], best[i][j + 1]]
            if (i, j) not in banned and durations_match(wavs[i], rows[j]):
                options.append(_add(pair_score(wavs[i], rows[j]), best[i + 1][j + 1]))
            best[i][j] = max(options)
    pairs = []
    i = j = 0
    while i < m and j < k:
        if (
            (i, j) not in banned
            and durations_match(wavs[i], rows[j])
            and best[i][j] == _add(pair_score(wavs[i], rows[j]), best[i + 1][j + 1])
        ):
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif best[i][j] == best[i + 1][j]:
            i += 1
        else:
            j += 1
    return best[0][0], pairs


def assign_class(
    vessel_class: str, wavs: list[Wav], rows: list[MetaRow]
) -> list[tuple[Wav, MetaRow, str]]:
    """Assign one metafile row to each WAV of a class. Returns (wav, row, match_source)."""
    wavs = sorted(wavs, key=lambda w: w.number)
    blocks = sorted({r.block for r in rows})
    results = []
    for block in blocks:
        block_rows = [r for r in rows if r.block == block]
        score, pairs = align(wavs, block_rows)
        results.append((score, block, block_rows, pairs))
    best_score = max(r[0] for r in results)
    winners = [r for r in results if r[0] == best_score]
    if len(winners) > 1:
        # Also when no block matches any duration (score 0): taking the first block would be
        # a silent guess (D-010).
        raise AlignmentError(
            f"{vessel_class}: blocks {[w[1] for w in winners]} align equally well"
        )
    score, block, block_rows, pairs = winners[0]

    # The optimum must be unique: forbidding any chosen pair must make the score worse.
    for pair in pairs:
        alt_score, _ = align(wavs, block_rows, banned=frozenset({pair}))
        if alt_score == score:
            wav, row = wavs[pair[0]], block_rows[pair[1]]
            raise AlignmentError(
                f"{vessel_class}: ambiguous alignment for WAV {wav.number} "
                f"(line {row.line} is not the only optimal choice)"
            )

    paired = {i: block_rows[j] for i, j in pairs}
    assigned = []
    for i, wav in enumerate(wavs):
        if i in paired:
            row = paired[i]
            source = SOURCE_ID if row.id == wav.number else SOURCE_DURATION
        else:
            by_id = [r for r in block_rows if r.id == wav.number]
            if not by_id:
                raise AlignmentError(
                    f"{vessel_class}: WAV {wav.number} matches no duration and no id in block "
                    f"{block}"
                )
            if len(by_id) > 1:
                raise AlignmentError(
                    f"{vessel_class}: WAV {wav.number} matches no duration and id "
                    f"{wav.number} is repeated in block {block} "
                    f"(lines {[r.line for r in by_id]})"
                )
            row = by_id[0]
            source = SOURCE_ID_UNVERIFIED
        assigned.append((wav, row, source))

    used = Counter(row.line for _, row, _ in assigned)
    repeated = sorted(line for line, n in used.items() if n > 1)
    if repeated:
        raise AlignmentError(f"{vessel_class}: metafile lines used twice: {repeated}")
    return assigned


# --------------------------------------------------------------------------- table


def find_metafile(class_dir: Path) -> Path:
    found = sorted(class_dir.glob("*-metafile"))
    if len(found) != 1:
        raise MetafileError(f"{class_dir}: expected one *-metafile, found {len(found)}")
    return found[0]


def build_table(data_dir: Path) -> list[dict[str, str]]:
    class_dirs = sorted(
        p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    table = []
    for class_dir in class_dirs:
        wav_paths = [p for p in class_dir.iterdir() if p.suffix.lower() == ".wav"]
        if not wav_paths:
            continue
        vessel_class = class_dir.name
        metafile = find_metafile(class_dir)
        rows = parse_metafile(metafile)
        wavs = [read_wav_header(p) for p in wav_paths]
        for wav, row, source in assign_class(vessel_class, wavs, rows):
            diff = wav.duration_s - row.duration_s
            table.append(
                {
                    "wav_path": wav.path.relative_to(data_dir).as_posix(),
                    "vessel_class": vessel_class,
                    "wav_number": str(wav.number),
                    "sha256": sha256_of(wav.path),
                    "sample_rate": str(wav.sample_rate),
                    "n_samples": str(wav.n_samples),
                    "wav_duration_s": f"{wav.duration_s:.6f}",
                    "metafile": metafile.relative_to(data_dir).as_posix(),
                    "metafile_line": str(row.line),
                    "metafile_block": str(row.block),
                    "metafile_id": str(row.id),
                    "ais_code": row.ais_code,
                    "vessel_name_raw": row.name_raw,
                    "vessel_group": vessel_group(vessel_class, row.name_raw),
                    "date": row.date,
                    "time": row.time,
                    "metafile_duration_s": str(row.duration_s),
                    "field7": row.field7,
                    "match_source": source,
                    "duration_match": str(abs(diff) <= DURATION_TOLERANCE_S).lower(),
                    "duration_diff_s": f"{diff:.6f}",
                }
            )
    if not table:
        raise FileNotFoundError(f"No WAV files found under {data_dir}")
    table.sort(key=lambda r: (r["vessel_class"], int(r["wav_number"])))
    return table


def write_table(table: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)


def summarize(table: list[dict[str, str]]) -> str:
    lines = []
    sources = Counter(r["match_source"] for r in table)
    groups = {r["vessel_group"] for r in table}
    lines.append(f"{len(table)} WAV, {len(groups)} vessel groups")
    lines.append(
        "match_source: " + ", ".join(f"{k}={sources[k]}" for k in sorted(sources))
    )
    for vessel_class in sorted({r["vessel_class"] for r in table}):
        rows = [r for r in table if r["vessel_class"] == vessel_class]
        n_groups = len({r["vessel_group"] for r in rows})
        lines.append(f"  {vessel_class}: {len(rows)} WAV, {n_groups} groups")
    for r in table:
        if r["match_source"] == SOURCE_ID_UNVERIFIED:
            lines.append(
                f"  unverified: {r['wav_path']} ({r['wav_duration_s']} s) -> line "
                f"{r['metafile_line']} {r['vessel_group']} ({r['metafile_duration_s']} s)"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    table = build_table(args.data_dir)
    write_table(table, args.output)
    print(summarize(table))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
