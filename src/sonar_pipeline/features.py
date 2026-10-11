"""Features (D-015): every recording of split.csv -> preprocessed 3 s segments -> LOFAR and DEMON.

Writes to the output folder:
- ``features_<split>.npz``: ``segment_id``, ``lofar`` (N, 11, 1001), ``demon`` (N, 301) as float32,
  and the frequency axes ``lofar_hz`` and ``demon_hz``.
- ``index.csv``: one row per segment, in split.csv order (LF, UTF-8).
- ``features_meta.json``: parameters, SHA-256 of split.csv, code hash and library versions.

Nothing is random: the same inputs give the same arrays, index and metadata.
"""

import csv
import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy

from sonar_pipeline import params
from sonar_pipeline.demon import demon, demon_frequencies
from sonar_pipeline.lofar import lofar, lofar_frequencies
from sonar_pipeline.preprocess import load_wav, preprocess, segment_length

SPLITS = ("train", "val", "test")
INDEX_FIELDS = [
    "segment_id",
    "wav_path",
    "vessel_class",
    "vessel_group",
    "split",
    "segment_index",
    "start_s",
]
PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass
class FeatureSummary:
    counts: Counter  # (vessel_class, split) -> number of segments
    n_recordings: int
    output_dir: Path
    figures: list[Path]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_sha256() -> str:
    """Hash of the package sources, so the metadata identifies the code even without git."""
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def project_commit() -> str | None:
    """Git commit of the project when git and the repository are available (not in the image)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(PACKAGE_DIR), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def read_split(split_csv: Path) -> list[dict]:
    with split_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{split_csv} has no rows")
    unknown = sorted({r["split"] for r in rows} - set(SPLITS))
    if unknown:
        raise ValueError(f"{split_csv}: unknown split values {unknown}")
    return rows


def metadata(split_csv: Path) -> dict:
    signal_params = {
        name: getattr(params, name) for name in dir(params) if name.isupper()
    }
    return {
        "decision": "D-015",
        "params": signal_params,
        "split_csv": split_csv.name,
        "split_sha256": sha256_of(split_csv),
        "project_commit": project_commit(),
        "code_sha256": code_sha256(),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
    }


def example_recordings(index: list[dict]) -> dict[str, str]:
    """The longest recording of every class (first in split.csv order on ties)."""
    n_segments = Counter(r["wav_path"] for r in index)
    chosen = {}
    for r in index:
        best = chosen.get(r["vessel_class"])
        if best is None or n_segments[r["wav_path"]] > n_segments[best]:
            chosen[r["vessel_class"]] = r["wav_path"]
    return chosen


def write_figures(
    index: list[dict], lofar_by_id: dict, demon_by_id: dict, figures_dir: Path
) -> list[Path]:
    from sonar_pipeline.figures import plot_recording

    written = []
    for vessel_class, wav_path in sorted(example_recordings(index).items()):
        rows = [r for r in index if r["wav_path"] == wav_path]
        ids = [r["segment_id"] for r in rows]
        demon_mean = np.mean([demon_by_id[i] for i in ids], axis=0)
        title = (
            f"{vessel_class} · {wav_path} · {rows[0]['vessel_group'].split('/', 1)[1]} "
            f"· {rows[0]['split']} · {len(ids)} segmentos de 3 s"
        )
        written.append(
            plot_recording(
                np.stack([lofar_by_id[i] for i in ids]),
                lofar_frequencies(),
                demon_mean / demon_mean.max(),
                demon_frequencies(segment_length()),
                title,
                figures_dir / f"example_{vessel_class.lower()}.png",
            )
        )
    return written


def run_features(
    data_dir: Path, split_csv: Path, output_dir: Path, figures_dir: Path | None = None
) -> FeatureSummary:
    rows = read_split(split_csv)
    missing = [r["wav_path"] for r in rows if not (data_dir / r["wav_path"]).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} WAV files of {split_csv} not found under {data_dir}, "
            f"e.g. {missing[0]}"
        )

    seg_seconds = segment_length() / params.SAMPLE_RATE
    index = []
    lofars = {s: [] for s in SPLITS}
    demons = {s: [] for s in SPLITS}
    ids = {s: [] for s in SPLITS}
    lofar_by_id, demon_by_id = {}, {}
    for row in rows:
        segments = preprocess(load_wav(data_dir / row["wav_path"]))
        for i, seg in enumerate(segments):
            segment_id = len(index)
            index.append(
                {
                    "segment_id": segment_id,
                    "wav_path": row["wav_path"],
                    "vessel_class": row["vessel_class"],
                    "vessel_group": row["vessel_group"],
                    "split": row["split"],
                    "segment_index": i,
                    "start_s": f"{i * seg_seconds:.3f}",
                }
            )
            ids[row["split"]].append(segment_id)
            lofar_by_id[segment_id] = lofar(seg).astype(np.float32)
            demon_by_id[segment_id] = demon(seg).astype(np.float32)
            lofars[row["split"]].append(lofar_by_id[segment_id])
            demons[row["split"]].append(demon_by_id[segment_id])

    output_dir.mkdir(parents=True, exist_ok=True)
    n_samples = segment_length()
    for split in SPLITS:
        if not ids[split]:
            continue
        np.savez(
            output_dir / f"features_{split}.npz",
            segment_id=np.array(ids[split], dtype=np.int64),
            lofar=np.stack(lofars[split]),
            demon=np.stack(demons[split]),
            lofar_hz=lofar_frequencies(),
            demon_hz=demon_frequencies(n_samples),
        )
    with (output_dir / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(index)
    meta = metadata(split_csv)
    (output_dir / "features_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    figures = []
    if figures_dir is not None:
        figures = write_figures(index, lofar_by_id, demon_by_id, figures_dir)

    counts = Counter((r["vessel_class"], r["split"]) for r in index)
    return FeatureSummary(
        counts=counts, n_recordings=len(rows), output_dir=output_dir, figures=figures
    )
