"""Metrics tests (F3-T04): small cases whose answer is worked out by hand."""

import json

import numpy as np
import pytest

from sonar_pipeline.dataset import CLASSES
from sonar_pipeline.metrics import (
    bootstrap_f1,
    confusion,
    evaluate_split,
    f1_scores,
    features_sha256,
    linear_model_cost,
    pd_at_pfa,
    roc_one_vs_rest,
    stopwatch,
    torch_cost,
    write_metrics,
)


def one_hot_probs(pred, confidence=0.7):
    """Probabilities whose argmax is pred."""
    probs = np.full((len(pred), len(CLASSES)), (1 - confidence) / (len(CLASSES) - 1))
    probs[np.arange(len(pred)), pred] = confidence
    return probs


def test_perfect_predictions_give_f1_one():
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    macro, per_class = f1_scores(y, y)
    assert macro == 1.0 and per_class == [1.0] * 4


def test_f1_by_hand():
    # Class 0: TP 1, FN 1, FP 0 -> P 1, R 0.5, F1 2/3. Class 1: TP 2, FP 1 -> F1 0.8.
    # Class 2: TP 1 -> 1. Class 3: never true nor predicted -> 0 (counted in the macro).
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    macro, per_class = f1_scores(y_true, y_pred)
    assert per_class == pytest.approx([2 / 3, 0.8, 1.0, 0.0])
    assert macro == pytest.approx((2 / 3 + 0.8 + 1.0) / 4)


def test_confusion_counts_and_rows_normalised():
    counts, normalised = confusion(np.array([0, 0, 1, 1, 1]), np.array([0, 1, 1, 1, 0]))
    assert counts[0] == [1, 1, 0, 0] and counts[1] == [1, 2, 0, 0]
    assert normalised[1] == pytest.approx([1 / 3, 2 / 3, 0, 0])
    assert normalised[3] == [0, 0, 0, 0]  # a class without samples stays at 0


def test_pd_at_pfa_interpolates():
    assert pd_at_pfa(np.array([0.0, 0.02, 1.0]), np.array([0.0, 0.8, 1.0])) == 0.4


def test_roc_auc_by_hand():
    # Cargo scores: negatives 0.1, 0.4; positives 0.35, 0.8 -> 3 of 4 pairs ordered: AUC 0.75.
    y = np.array([1, 1, 0, 0])
    probs = np.zeros((4, 4))
    probs[:, 0] = [0.1, 0.4, 0.35, 0.8]
    probs[:, 1] = 1 - probs[:, 0]
    roc = roc_one_vs_rest(y, probs)
    assert roc["Cargo"]["auc"] == pytest.approx(0.75)
    assert roc["Passengership"]["auc"] == pytest.approx(0.75)
    assert roc["Tanker"] is None and roc["Tug"] is None  # no positives


def test_perfectly_separated_roc():
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    roc = roc_one_vs_rest(y, one_hot_probs(y))
    for name in CLASSES:
        assert roc[name]["auc"] == 1.0
        assert roc[name]["pd_at_pfa_1pct"] == 1.0
        assert roc[name]["pfa"][0] == 0.0 and roc[name]["pd"][-1] == 1.0


def test_bootstrap_resamples_recordings_not_segments():
    # One recording per class: resampling recordings always gives the same data, so the
    # interval collapses on the point value. Resampling segments would make it vary.
    y_true = np.repeat([0, 1, 2, 3], 10)
    y_pred = y_true.copy()
    y_pred[[0, 1, 2]] = 1  # 3 Cargo segments wrong
    groups = np.repeat(["c.wav", "p.wav", "t.wav", "u.wav"], 10)
    ci = bootstrap_f1(y_true, y_pred, groups, n_samples=200)
    macro, _ = f1_scores(y_true, y_pred)
    assert ci["f1_macro"] == pytest.approx([macro, macro])


def test_bootstrap_interval_contains_the_point_and_is_reproducible():
    rng = np.random.default_rng(1)
    y_true = np.repeat([0, 1, 2, 3], 30)
    groups = np.array(
        [f"{k}-{i // 5}.wav" for i, k in enumerate(y_true)]
    )  # 6 per class
    y_pred = np.where(rng.random(120) < 0.7, y_true, rng.integers(0, 4, 120))
    a = bootstrap_f1(y_true, y_pred, groups, n_samples=300, seed=3)
    b = bootstrap_f1(y_true, y_pred, groups, n_samples=300, seed=3)
    assert a == b
    macro, _ = f1_scores(y_true, y_pred)
    assert a["f1_macro"][0] < macro < a["f1_macro"][1]


def test_evaluate_split_has_both_levels():
    labels = np.repeat([0, 1, 2, 3], 4)
    wav = np.array([f"{k}-{i % 2}.wav" for i, k in enumerate(labels)])
    result = evaluate_split(one_hot_probs(labels), labels, wav, n_bootstrap=50)
    assert result["segment"]["n"] == 16 and result["recording"]["n"] == 8
    assert result["recording"]["f1_macro"] == 1.0
    assert set(result["segment"]["f1_per_class"]) == set(CLASSES)


def test_torch_cost_counts_parameters_and_flops():
    import torch

    model = torch.nn.Linear(10, 3)
    cost = torch_cost(model, (torch.zeros(1, 10),))
    assert cost["parameters"] == 10 * 3 + 3
    assert cost["flops_per_segment"] == 2 * 10 * 3  # one multiply-add per weight
    assert model.training  # the mode is restored


def test_linear_model_cost():
    assert linear_model_cost(1302) == {
        "parameters": 1302 * 4 + 4,
        "flops_per_segment": 2 * 1302 + 2 * 1302 * 4,
    }


def test_stopwatch_measures_time():
    with stopwatch() as t:
        sum(range(1000))
    assert 0 <= t["seconds"] < 5


def test_write_metrics_is_strict_json(tmp_path):
    path = write_metrics(
        tmp_path / "out" / "metrics.json",
        {
            "b": np.float32(0.5),
            "a": np.arange(3),
            "nan": float("nan"),
            "n": np.int64(2),
        },
    )
    raw = path.read_bytes()
    assert b"\r" not in raw and b"NaN" not in raw
    data = json.loads(raw)
    assert data == {"schema": 1, "a": [0, 1, 2], "b": 0.5, "n": 2, "nan": None}
    assert list(data) == sorted(data)


def test_features_sha256_depends_on_content_only(tmp_path):
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
        np.savez(tmp_path / name / "features_train.npz", lofar=np.ones((2, 3)))
    first = features_sha256(tmp_path / "one")
    assert first == features_sha256(
        tmp_path / "two"
    )  # same arrays, different zip bytes
    np.savez(tmp_path / "two" / "features_train.npz", lofar=np.zeros((2, 3)))
    assert first != features_sha256(tmp_path / "two")


def test_traceability_of_the_real_features():
    from pathlib import Path

    from sonar_pipeline.metrics import traceability

    processed = Path(__file__).resolve().parents[1] / "datasets" / "processed"
    if not (processed / "features_meta.json").is_file():
        pytest.skip(f"{processed} has no features: run `features` first (README)")
    trace = traceability(processed)
    assert trace["deepship_commit"] == "77725726a33d6aef4ce1d0fe70f67c0bbee4b5b6"
    assert trace["split_sha256"].startswith("51564cf1")
    assert set(trace["features_sha256"]) == {"train", "val", "test"}
