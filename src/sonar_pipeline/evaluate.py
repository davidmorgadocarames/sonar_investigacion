"""Evaluate (F3-T09, D-018): score the frozen models on test, once, and write metrics.json.

The weights are the versioned ones in models/; their SHA-256 goes into metrics.json so the
evaluated models are identified exactly. CPU is the reference device. Nothing is trained here.
The val scores, cost and training times come from each model's JSON; the DT-005 duration check
and the permutation test come from outputs/results when they exist.
"""

import hashlib
import json
from pathlib import Path

from sonar_pipeline.dataset import CLASSES, baseline_features, load_split
from sonar_pipeline.features import code_sha256
from sonar_pipeline.metrics import (
    evaluate_split,
    hardware,
    stopwatch,
    traceability,
    write_metrics,
)

MODELS = ("baseline", "cnn", "cnn_aug")
# Test evaluation these versioned weights belong to (F3-T12: the second and last one). Change it
# only with David's approval: every new evaluation is one more look at test.
EVALUATION = 2
# Sections copied into metrics.json from their own JSON files in the results folder.
SECTIONS = {
    "dt005": "dt005_duration.json",
    "permutation_test": "permutation_test.json",
    "interpretability": "interpretability.json",
}
WEIGHT_FILES = {"baseline": "baseline.npz", "cnn": "cnn.pt", "cnn_aug": "cnn_aug.pt"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def test_probabilities(name: str, weights_path: Path, test, device_name: str) -> tuple:
    """Probabilities of every test segment and the inference time (s)."""
    if name == "baseline":
        from sonar_pipeline.baseline import load_baseline, predict_proba

        weights = load_baseline(weights_path)
        x = baseline_features(test)
        with stopwatch() as elapsed:
            probabilities = predict_proba(weights, x)
        return probabilities, elapsed["seconds"], "cpu"

    import torch

    from sonar_pipeline.cnn import TORCH_THREADS, load_cnn, predict_proba, select_device

    torch.set_num_threads(TORCH_THREADS)  # same as training (D-018 adjustment 2)
    device = select_device(device_name)
    model = load_cnn(weights_path, device)
    lofar = torch.as_tensor(test.lofar, device=device)
    demon = torch.as_tensor(test.demon, device=device)
    with stopwatch() as elapsed:
        probabilities = predict_proba(model, lofar, demon)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return probabilities, elapsed["seconds"], str(device)


def evaluate_models(
    processed_dir: Path,
    models_dir: Path,
    results_dir: Path,
    figures_dir: Path | None = None,
    device_name: str = "cpu",
    n_bootstrap: int = 2000,
) -> dict:
    missing = [
        WEIGHT_FILES[m] for m in MODELS if not (models_dir / WEIGHT_FILES[m]).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"weights not found in {models_dir}: {missing}")
    test = load_split(processed_dir, "test")

    models = {}
    for name in MODELS:
        weights_path = models_dir / WEIGHT_FILES[name]
        trained = read_json(models_dir / f"{name}.json")
        probabilities, seconds, device = test_probabilities(
            name, weights_path, test, device_name
        )
        entry = {
            "description": trained["description"],
            "config": trained["config"],
            "cost": trained["cost"],
            "training": trained["training"],
            "inference": {
                "val": trained["inference"],
                "test": {"seconds": seconds, "segments": len(test), "device": device},
            },
            "val": trained["val"],
            "test": evaluate_split(
                probabilities, test.labels, test.wav_path, n_bootstrap
            ),
            "weights": weights_path.name,
            "weights_sha256": sha256_file(weights_path),
            "training_hardware": trained.get("hardware"),
            "training_code_sha256": trained.get("code_sha256"),
        }
        for key in ("grouped_cv", "comparison"):
            if key in trained:
                entry[key] = trained[key]
        models[name] = entry

    payload = {
        "description": "F3 results: test scored with the frozen, versioned models (D-018)",
        "evaluation": EVALUATION,
        "classes": list(CLASSES),
        "traceability": traceability(processed_dir),
        "hardware": hardware(device_name),
        "code_sha256": code_sha256(),
        "models": models,
        **{key: read_json(results_dir / name) for key, name in SECTIONS.items()},
    }
    write_metrics(results_dir / "metrics.json", payload)

    if figures_dir is not None:
        from sonar_pipeline.figures import plot_results

        payload["figures"] = [str(p) for p in plot_results(models, figures_dir)]
    return payload


def refresh_metrics(results_dir: Path) -> dict:
    """Reload the DT-005, permutation and interpretability sections into metrics.json and set
    the evaluation number, without loading or scoring test."""
    metrics = read_json(results_dir / "metrics.json")
    if metrics is None:
        raise FileNotFoundError(
            f"{results_dir / 'metrics.json'} not found: run evaluate first"
        )
    metrics.pop("schema", None)
    for key, name in SECTIONS.items():
        metrics[key] = read_json(results_dir / name)
    metrics["evaluation"] = EVALUATION
    write_metrics(results_dir / "metrics.json", metrics)
    return metrics


def plot_from_metrics(results_dir: Path, figures_dir: Path) -> list[Path]:
    """Redraw the result figures from metrics.json, without scoring test again."""
    from sonar_pipeline.figures import plot_results

    metrics = read_json(results_dir / "metrics.json")
    if metrics is None:
        raise FileNotFoundError(
            f"{results_dir / 'metrics.json'} not found: run evaluate first"
        )
    return plot_results(metrics["models"], figures_dir)
