"""Baseline (F3-T06, D-018): logistic regression on the time-mean LOFAR and the DEMON.

- C is chosen on val (segment-level macro F1; ties go to the smaller C, more regularised).
- The chosen model is saved without pickle: ``models/baseline.npz`` (scaler mean and scale,
  coefficients, intercepts) plus ``models/baseline.json`` (configuration, val scores, cost,
  timing, hardware, traceability and the vessel-grouped cross-validation).
- Cross-validation: stratified and grouped by vessel, 5 folds over train + val.
- Permutation test (a separate step, after the model is committed): class labels are shuffled
  between vessel groups and the same cross-validation is repeated.

Test is never loaded here.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sonar_pipeline.dataset import CLASSES, baseline_features, load_split
from sonar_pipeline.metrics import (
    evaluate_split,
    f1_scores,
    hardware,
    linear_model_cost,
    stopwatch,
    traceability,
    write_metrics,
)

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
CV_FOLDS = 5
PERMUTATIONS = 200
SEED = 0
MAX_ITER = 5000


def make_model(c: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=MAX_ITER),
    )


def present_f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F1 over the classes present in y_true (a fold may lack a class, e.g. Tug)."""
    macro, _ = f1_scores(y_true, y_pred, sorted(set(y_true.tolist())))
    return macro


def grouped_cv(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, c: float, seed: int = SEED
) -> list[float]:
    """Macro F1 of every fold of a stratified, vessel-grouped cross-validation."""
    folds = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    scores = []
    for train_rows, test_rows in folds.split(x, y, groups):
        model = make_model(c).fit(x[train_rows], y[train_rows])
        scores.append(present_f1_macro(y[test_rows], model.predict(x[test_rows])))
    return scores


def save_baseline(model, path: Path, c: float) -> Path:
    scaler, logistic = model[0], model[-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=scaler.mean_,
        scale=scaler.scale_,
        coef=logistic.coef_,
        intercept=logistic.intercept_,
        c=np.array(c),
        classes=np.array(CLASSES),
    )
    return path


def load_baseline(path: Path) -> dict:
    with np.load(path) as npz:  # plain arrays, no pickle
        weights = {key: npz[key] for key in npz.files}
    if tuple(weights["classes"].tolist()) != CLASSES:
        raise ValueError(f"{path}: classes {weights['classes']} differ from {CLASSES}")
    return weights


def predict_proba(weights: dict, x: np.ndarray) -> np.ndarray:
    """Softmax of the standardised linear scores, as LogisticRegression.predict_proba."""
    logits = ((x - weights["mean"]) / weights["scale"]) @ weights["coef"].T
    logits = logits + weights["intercept"]
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def train_baseline(
    processed_dir: Path,
    models_dir: Path,
    n_bootstrap: int = 2000,
    c_grid: tuple[float, ...] = C_GRID,
) -> dict:
    train = load_split(processed_dir, "train")
    val = load_split(processed_dir, "val")
    x_train, x_val = baseline_features(train), baseline_features(val)

    search = []
    best = None
    with stopwatch() as search_time:
        for c in c_grid:
            with stopwatch() as fit_time:
                model = make_model(c).fit(x_train, train.labels)
            score = present_f1_macro(val.labels, model.predict(x_val))
            search.append(
                {"c": c, "val_f1_macro": score, "fit_seconds": fit_time["seconds"]}
            )
            if best is None or score > best[0]:  # strict: ties keep the smaller C
                best = (score, c, model, fit_time["seconds"])
    _, c, model, fit_seconds = best

    with stopwatch() as inference_time:
        probabilities = model.predict_proba(x_val)

    x_all = np.vstack([x_train, x_val])
    y_all = np.concatenate([train.labels, val.labels])
    groups_all = np.concatenate([train.vessel_group, val.vessel_group])
    with stopwatch() as cv_time:
        folds = grouped_cv(x_all, y_all, groups_all, c)

    weights_path = save_baseline(model, models_dir / "baseline.npz", c)
    report = {
        "model": "baseline",
        "description": "standardised logistic regression on time-mean LOFAR + DEMON (D-018)",
        "config": {
            "c": c,
            "c_grid": list(c_grid),
            "class_weight": "balanced",
            "max_iter": MAX_ITER,
            "n_features": int(x_train.shape[1]),
            "selection": "segment-level macro F1 on val; ties to the smaller C",
        },
        "cost": linear_model_cost(x_train.shape[1]),
        "training": {
            "seconds": fit_seconds,
            "search_seconds": search_time["seconds"],
            "segments": len(train),
            "device": "cpu",
            "search": search,
        },
        "inference": {"seconds": inference_time["seconds"], "segments": len(val)},
        "val": evaluate_split(probabilities, val.labels, val.wav_path, n_bootstrap),
        "grouped_cv": {
            "splits": "train + val",
            "folds": CV_FOLDS,
            "seed": SEED,
            "f1_macro": folds,
            "mean": float(np.mean(folds)),
            "std": float(np.std(folds)),
            "seconds": cv_time["seconds"],
        },
        "hardware": hardware("cpu"),
        "traceability": traceability(processed_dir),
        "weights": weights_path.name,
    }
    write_metrics(models_dir / "baseline.json", report)
    return report


def shuffle_by_group(
    y: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle the classes between vessel groups: every segment of a group keeps one label,
    and the number of groups per class stays the same."""
    names = list(dict.fromkeys(groups.tolist()))
    group_class = np.array([y[groups == g][0] for g in names])
    row_group = np.array([names.index(g) for g in groups])
    return rng.permutation(group_class)[row_group]


def permutation_test(
    processed_dir: Path,
    models_dir: Path,
    output: Path,
    n_permutations: int = PERMUTATIONS,
    seed: int = SEED,
) -> dict:
    """Shuffle classes between vessel groups and repeat the grouped cross-validation.

    p = (1 + number of permutations scoring >= the real labels) / (1 + n_permutations).
    Uses the C saved in models/baseline.json, so the model must be trained first.
    """
    c = json.loads((models_dir / "baseline.json").read_text(encoding="utf-8"))[
        "config"
    ]["c"]
    train = load_split(processed_dir, "train")
    val = load_split(processed_dir, "val")
    x = np.vstack([baseline_features(train), baseline_features(val)])
    y = np.concatenate([train.labels, val.labels])
    groups = np.concatenate([train.vessel_group, val.vessel_group])

    rng = np.random.default_rng(seed)
    with stopwatch() as elapsed:
        observed = float(np.mean(grouped_cv(x, y, groups, c)))
        null = []
        for _ in range(n_permutations):
            shuffled = shuffle_by_group(y, groups, rng)
            null.append(float(np.mean(grouped_cv(x, shuffled, groups, c))))
    exceed = int(np.sum(np.array(null) >= observed))
    result = {
        "model": "baseline",
        "c": c,
        "unit": "vessel group (labels shuffled between groups)",
        "n_permutations": n_permutations,
        "seed": seed,
        "observed_f1_macro": observed,
        "null_f1_macro": null,
        "null_mean": float(np.mean(null)),
        "null_p95": float(np.percentile(null, 95)),
        "exceed": exceed,
        "p_value": (1 + exceed) / (1 + n_permutations),
        "seconds": elapsed["seconds"],
    }
    write_metrics(output, result)
    return result
