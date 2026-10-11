"""Metrics and metrics.json (D-018).

Scores are computed per segment and per recording (mean of segment probabilities). Confidence
intervals come from a bootstrap that resamples recordings within each class, never segments,
so every class stays present and segments of one recording move together.

Layout of ``outputs/results/metrics.json`` (filled by F3-T05 to F3-T10):

    {
      "schema": 1,
      "traceability": {split_sha256, features_sha256, code_sha256, project_commit, ...},
      "hardware": {platform, cpu, threads, device, gpu, versions},
      "models": {
        "<name>": {
          "cost": {parameters, flops_per_segment},
          "training": {seconds, device, seeds, ...},
          "inference": {seconds, segments},
          "val": {"segment": SCORES, "recording": SCORES},
          "test": {"segment": SCORES, "recording": SCORES}
        }
      },
      "dt005": {...}, "permutation_test": {...}, "interpretability": {...}
    }

    SCORES = {n, f1_macro, f1_per_class, ci95 {f1_macro, f1_per_class}, confusion,
              confusion_normalized, roc {class: {pfa, pd, auc, pd_at_pfa_1pct}}}
"""

import hashlib
import json
import math
import os
import platform
import time
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve

from sonar_pipeline.dataset import CLASSES, aggregate_by_recording

SCHEMA_VERSION = 1
BOOTSTRAP_SAMPLES = 2000
CI_LEVEL = 0.95
PFA_TARGET = 0.01
PACKAGES = ("numpy", "scipy", "scikit-learn", "torch", "matplotlib")


# --------------------------------------------------------------------------- scores


def f1_scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, list[float]]:
    """Macro F1 and F1 of every class, over all classes (absent predictions give 0)."""
    labels = list(range(len(CLASSES)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return float(np.mean(per_class)), [float(v) for v in per_class]


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[list, list]:
    """Counts (rows: true class, columns: predicted) and rows normalised to sum 1."""
    counts = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    totals = counts.sum(axis=1, keepdims=True)
    normalised = np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)
    return counts.tolist(), normalised.tolist()


def pd_at_pfa(pfa: np.ndarray, pd: np.ndarray, target: float = PFA_TARGET) -> float:
    """Probability of detection at a false-alarm probability, linearly interpolated."""
    return float(np.interp(target, pfa, pd))


def roc_one_vs_rest(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    """ROC of every class against the rest: Pfa (false positive rate), Pd, AUC, Pd at 1 %."""
    result = {}
    for k, name in enumerate(CLASSES):
        positive = y_true == k
        if positive.all() or not positive.any():
            result[name] = None  # undefined without both positives and negatives
            continue
        pfa, pd, _ = roc_curve(positive, probabilities[:, k])
        result[name] = {
            "pfa": pfa.tolist(),
            "pd": pd.tolist(),
            "auc": float(roc_auc_score(positive, probabilities[:, k])),
            "pd_at_pfa_1pct": pd_at_pfa(pfa, pd),
        }
    return result


def bootstrap_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
    level: float = CI_LEVEL,
) -> dict:
    """Percentile interval of macro and per-class F1, resampling groups within each class.

    groups: the recording of every row. Each resample draws, for every class, as many of its
    recordings as it has (with replacement) and keeps all their rows.
    """
    rng = np.random.default_rng(seed)
    rows_of = {g: np.flatnonzero(groups == g) for g in dict.fromkeys(groups.tolist())}
    by_class = {}
    for g, rows in rows_of.items():
        by_class.setdefault(int(y_true[rows[0]]), []).append(g)
    macro, per_class = [], []
    for _ in range(n_samples):
        chosen = []
        for k in sorted(by_class):
            members = by_class[k]
            picks = rng.integers(0, len(members), len(members))
            chosen.extend(rows_of[members[i]] for i in picks)
        rows = np.concatenate(chosen)
        m, pc = f1_scores(y_true[rows], y_pred[rows])
        macro.append(m)
        per_class.append(pc)
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    per_class = np.array(per_class)
    return {
        "f1_macro": [float(np.percentile(macro, lo)), float(np.percentile(macro, hi))],
        "f1_per_class": {
            name: [
                float(np.percentile(per_class[:, k], lo)),
                float(np.percentile(per_class[:, k], hi)),
            ]
            for k, name in enumerate(CLASSES)
        },
    }


def scores(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> dict:
    """All the scores of one level (segment or recording) in metrics.json form."""
    y_pred = probabilities.argmax(axis=1)
    macro, per_class = f1_scores(y_true, y_pred)
    counts, normalised = confusion(y_true, y_pred)
    return {
        "n": len(y_true),
        "f1_macro": macro,
        "f1_per_class": dict(zip(CLASSES, per_class, strict=True)),
        "ci95": bootstrap_f1(y_true, y_pred, groups, n_bootstrap, seed),
        "confusion": counts,
        "confusion_normalized": normalised,
        "roc": roc_one_vs_rest(y_true, probabilities),
    }


def evaluate_split(
    probabilities: np.ndarray,
    labels: np.ndarray,
    wav_path: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> dict:
    """Segment-level and recording-level scores of one split."""
    rec = aggregate_by_recording(probabilities, wav_path, labels)
    return {
        "segment": scores(probabilities, labels, wav_path, n_bootstrap, seed),
        "recording": scores(
            rec.probabilities, rec.labels, rec.wav_path, n_bootstrap, seed
        ),
    }


# --------------------------------------------------------------------------- cost


def torch_cost(model, example_inputs: tuple) -> dict:
    """Trainable parameters and FLOPs of one forward pass on a single segment."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(*example_inputs)
    model.train(was_training)
    return {
        "parameters": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "flops_per_segment": int(counter.get_total_flops()),
    }


def linear_model_cost(n_features: int, n_classes: int = len(CLASSES)) -> dict:
    """Logistic regression with standardisation: weights + biases; scaling and matmul FLOPs."""
    return {
        "parameters": n_features * n_classes + n_classes,
        "flops_per_segment": 2 * n_features + 2 * n_features * n_classes,
    }


@contextmanager
def stopwatch():
    """`with stopwatch() as t: ...` then t["seconds"] holds the wall-clock time."""
    timing = {}
    start = time.perf_counter()
    try:
        yield timing
    finally:
        timing["seconds"] = time.perf_counter() - start


# --------------------------------------------------------------------------- environment


def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def package_versions() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def hardware(device: str = "cpu") -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        "device": device,
        "gpu": None,
        "versions": package_versions(),
    }
    try:
        import torch

        info["torch_threads"] = torch.get_num_threads()
        if device.startswith("cuda") and torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(torch.device(device))
    except ImportError:
        info["torch_threads"] = None
    return info


def features_sha256(processed_dir: Path) -> dict:
    """Content hash of every split's arrays (the .npz zip bytes change between runs)."""
    hashes = {}
    for path in sorted(processed_dir.glob("features_*.npz")):
        digest = hashlib.sha256()
        with np.load(path) as npz:
            for key in sorted(npz.files):
                array = np.ascontiguousarray(npz[key])
                digest.update(key.encode())
                digest.update(str(array.dtype).encode())
                digest.update(str(array.shape).encode())
                digest.update(array.tobytes())
        hashes[path.stem.removeprefix("features_")] = digest.hexdigest()
    return hashes


def traceability(processed_dir: Path) -> dict:
    """Where the features came from (features_meta.json) plus a content hash of the arrays."""
    meta = json.loads(
        (processed_dir / "features_meta.json").read_text(encoding="utf-8")
    )
    keys = ("split_sha256", "vessels_sha256", "deepship_commit", "project_commit")
    return {
        **{key: meta.get(key) for key in keys},
        "features_code_sha256": meta.get("code_sha256"),
        "features_sha256": features_sha256(processed_dir),
        "index_sha256": hashlib.sha256(
            (processed_dir / "index.csv").read_bytes()
        ).hexdigest(),
    }


# --------------------------------------------------------------------------- json


def _plain(value):
    """numpy and non-finite values -> plain JSON (NaN and infinities become null)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_metrics(path: Path, payload: dict) -> Path:
    """Write metrics.json: sorted keys, UTF-8, LF, strict JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema": SCHEMA_VERSION, **_plain(payload)}
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return path
