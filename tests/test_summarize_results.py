"""tools/summarize_results.py (D-020): summary of a results folder, without scoring test."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from test_evaluate import run, trained  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "summarize_results", ROOT / "tools" / "summarize_results.py"
)
summarize_results = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = summarize_results
spec.loader.exec_module(summarize_results)


def test_summary_of_a_results_folder(trained, tmp_path, monkeypatch):  # noqa: F811
    folder, models = trained
    assert run(folder, models, tmp_path) == 0
    results = tmp_path / "r"

    def no_scoring(*args, **kwargs):
        raise AssertionError("the summary must not score test")

    monkeypatch.setattr("sonar_pipeline.evaluate.evaluate_split", no_scoring)
    out = tmp_path / "docs" / "eval9.json"
    assert summarize_results.main([str(results), str(out), "--evaluation", "9"]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["evaluation"] == 9
    metrics = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
    for name, entry in summary["models"].items():
        full = metrics["models"][name]
        assert entry["weights_sha256"] == full["weights_sha256"]
        assert entry["test"]["segment"]["f1_macro"] == round(
            full["test"]["segment"]["f1_macro"], 4
        )
        assert set(entry["test"]["within_era"]) == {"2016", "2017"}
    assert "val_f1_by_seed" in summary["models"]["cnn"]
    assert "grouped_cv" in summary["models"]["baseline"]
    assert "permutation_test" not in summary  # not run in this folder


def test_missing_metrics_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        summarize_results.summarize(tmp_path, 1)
