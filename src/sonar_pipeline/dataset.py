"""Model data (D-018): load one split of the features, labels, class weights and per-recording
aggregation.

A split is always loaded on its own and checked against index.csv: a segment whose index row
belongs to another split, or an index row missing from the .npz, is an error. Splits are never
mixed.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sonar_pipeline.features import SPLITS

# Fixed class order: label k is CLASSES[k] in every model, metric and figure.
CLASSES = ("Cargo", "Passengership", "Tanker", "Tug")

# Recording era of every class (``date`` in vessels.csv). Class and era coincide (DT-005), so
# only the comparisons inside one era are free of recording conditions (D-018 adjustment).
ERAS = {"2016": ("Passengership", "Tanker"), "2017": ("Cargo", "Tug")}


@dataclass
class SplitData:
    split: str
    segment_id: np.ndarray  # (N,) int64
    lofar: np.ndarray  # (N, n_frames, n_lofar_bins) float32, dB above the background
    demon: np.ndarray  # (N, n_demon_bins) float32
    labels: np.ndarray  # (N,) int64, index into CLASSES
    wav_path: np.ndarray  # (N,) recording of every segment
    vessel_group: np.ndarray  # (N,) vessel group of every segment
    lofar_hz: np.ndarray
    demon_hz: np.ndarray

    def __len__(self) -> int:
        return len(self.segment_id)


def read_index(processed_dir: Path) -> dict[int, dict]:
    with (processed_dir / "index.csv").open(newline="", encoding="utf-8") as f:
        return {int(r["segment_id"]): r for r in csv.DictReader(f)}


def load_split(processed_dir: Path, split: str) -> SplitData:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    index = read_index(processed_dir)
    with np.load(processed_dir / f"features_{split}.npz") as npz:
        arrays = {key: npz[key] for key in npz.files}

    ids = arrays["segment_id"]
    missing = [int(i) for i in ids if int(i) not in index]
    if missing:
        raise ValueError(
            f"features_{split}.npz: segments {missing[:5]} not in index.csv"
        )
    rows = [index[int(i)] for i in ids]
    foreign = sorted({r["split"] for r in rows} - {split})
    if foreign:
        raise ValueError(f"features_{split}.npz has segments of the {foreign} split")
    expected = {i for i, r in index.items() if r["split"] == split}
    if expected != {int(i) for i in ids}:
        raise ValueError(
            f"features_{split}.npz and index.csv disagree on the {split} split"
        )
    unknown = sorted({r["vessel_class"] for r in rows} - set(CLASSES))
    if unknown:
        raise ValueError(f"unknown classes {unknown}, expected {CLASSES}")
    n = len(ids)
    if not (len(arrays["lofar"]) == len(arrays["demon"]) == n):
        raise ValueError(f"features_{split}.npz: arrays of different lengths")

    return SplitData(
        split=split,
        segment_id=ids.astype(np.int64),
        lofar=arrays["lofar"],
        demon=arrays["demon"],
        labels=np.array(
            [CLASSES.index(r["vessel_class"]) for r in rows], dtype=np.int64
        ),
        wav_path=np.array([r["wav_path"] for r in rows]),
        vessel_group=np.array([r["vessel_group"] for r in rows]),
        lofar_hz=arrays["lofar_hz"],
        demon_hz=arrays["demon_hz"],
    )


def class_weights(labels: np.ndarray, n_classes: int = len(CLASSES)) -> np.ndarray:
    """Balanced weights n / (n_classes * count_k), as scikit-learn's class_weight="balanced".

    Every class must be present: a missing class cannot be weighted.
    """
    counts = np.bincount(labels, minlength=n_classes)
    if np.any(counts == 0):
        absent = [CLASSES[k] for k in np.flatnonzero(counts == 0)]
        raise ValueError(f"classes without samples: {absent}")
    return len(labels) / (n_classes * counts)


def baseline_features(data: SplitData) -> np.ndarray:
    """Baseline input (D-018): time-mean LOFAR (n_lofar_bins) next to DEMON (n_demon_bins)."""
    return np.hstack([data.lofar.mean(axis=1), data.demon]).astype(np.float64)


@dataclass
class RecordingScores:
    wav_path: np.ndarray  # (R,) in order of first appearance
    probabilities: np.ndarray  # (R, n_classes): mean of the segment probabilities
    labels: np.ndarray  # (R,)


def aggregate_by_recording(
    probabilities: np.ndarray, wav_path: np.ndarray, labels: np.ndarray
) -> RecordingScores:
    """One decision per recording: the mean of its segment probabilities (D-018)."""
    if not (len(probabilities) == len(wav_path) == len(labels)):
        raise ValueError("probabilities, wav_path and labels have different lengths")
    order = list(dict.fromkeys(wav_path.tolist()))
    rec_probs, rec_labels = [], []
    for path in order:
        mask = wav_path == path
        segment_labels = np.unique(labels[mask])
        if len(segment_labels) != 1:
            raise ValueError(f"{path}: segments with different labels {segment_labels}")
        rec_probs.append(probabilities[mask].mean(axis=0))
        rec_labels.append(segment_labels[0])
    return RecordingScores(
        wav_path=np.array(order),
        probabilities=np.array(rec_probs),
        labels=np.array(rec_labels, dtype=np.int64),
    )
