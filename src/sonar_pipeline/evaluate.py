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

    from sonar_pipeline.cnn import load_cnn, predict_proba, select_device

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
        "description": "F3 results: test scored once with the frozen models (D-018)",
        "classes": list(CLASSES),
        "traceability": traceability(processed_dir),
        "hardware": hardware(device_name),
        "code_sha256": code_sha256(),
        "models": models,
        "dt005": read_json(results_dir / "dt005_duration.json"),
        "permutation_test": read_json(results_dir / "permutation_test.json"),
    }
    write_metrics(results_dir / "metrics.json", payload)

    if figures_dir is not None:
        from sonar_pipeline.figures import plot_results

        payload["figures"] = [str(p) for p in plot_results(models, figures_dir)]
    return payload


def plot_from_metrics(results_dir: Path, figures_dir: Path) -> list[Path]:
    """Redraw the result figures from metrics.json, without scoring test again."""
    from sonar_pipeline.figures import plot_results

    metrics = read_json(results_dir / "metrics.json")
    if metrics is None:
        raise FileNotFoundError(
            f"{results_dir / 'metrics.json'} not found: run evaluate first"
        )
    return plot_results(metrics["models"], figures_dir)
