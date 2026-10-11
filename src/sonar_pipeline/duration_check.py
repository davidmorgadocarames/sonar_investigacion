"""DT-005 check (F3-T05, D-018): how well does the recording duration alone predict the class?

Class and recording era coincide in DeepShip, and so does duration (D-009). A logistic regression
that only sees the duration of each segment's recording is fitted on train and scored on val.
Test is never loaded. If this trivial model scores close to the real ones, they may be learning
the recording conditions instead of the vessel signature.
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sonar_pipeline.dataset import CLASSES, load_split
from sonar_pipeline.metrics import evaluate_split, stopwatch, write_metrics


def recording_durations(vessels_csv: Path) -> dict[str, float]:
    with vessels_csv.open(newline="", encoding="utf-8") as f:
        return {r["wav_path"]: float(r["wav_duration_s"]) for r in csv.DictReader(f)}


def duration_feature(wav_path: np.ndarray, durations: dict[str, float]) -> np.ndarray:
    missing = sorted(set(wav_path) - set(durations))
    if missing:
        raise ValueError(f"no duration for {missing[:3]} in vessels.csv")
    return np.array([[durations[p]] for p in wav_path])


def run_duration_check(
    processed_dir: Path, vessels_csv: Path, output: Path, n_bootstrap: int = 2000
) -> dict:
    durations = recording_durations(vessels_csv)
    train = load_split(processed_dir, "train")
    val = load_split(processed_dir, "val")

    model = make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
    )
    with stopwatch() as fit_time:
        model.fit(duration_feature(train.wav_path, durations), train.labels)
    if list(model.classes_) != list(range(len(CLASSES))):
        raise ValueError(f"train lacks some classes: {model.classes_}")
    probabilities = model.predict_proba(duration_feature(val.wav_path, durations))

    result = {
        "model": "logistic regression on the recording duration only (DT-005)",
        "feature": "wav_duration_s from vessels.csv",
        "training": {"seconds": fit_time["seconds"], "segments": len(train)},
        "val": evaluate_split(probabilities, val.labels, val.wav_path, n_bootstrap),
        "chance_f1_macro": 1 / len(CLASSES),
    }
    write_metrics(output, result)
    return result
