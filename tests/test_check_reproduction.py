"""tools/check_reproduction.py (D-023): does an evaluation reproduce the reference one?"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from test_evaluate import run, trained  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(
        f"{name}_tool", ROOT / "tools" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = load_tool("check_reproduction")
summarize = load_tool("summarize_results")


@pytest.fixture
def evaluated(trained, tmp_path):  # noqa: F811
    folder, models = trained
    assert run(folder, models, tmp_path) == 0
    metrics = tmp_path / "r" / "metrics.json"
    reference = tmp_path / "eval.json"
    summarize.main([str(tmp_path / "r"), str(reference), "--evaluation", "2"])
    return metrics, reference


def args(metrics, reference):
    return ["--metrics", str(metrics), "--reference", str(reference)]


def test_same_evaluation_reproduces(evaluated, capsys):
    assert check.main(args(*evaluated)) == 0
    assert "Reproduces" in capsys.readouterr().out


def test_a_change_beyond_the_tolerance_is_reported(evaluated, capsys):
    metrics, reference = evaluated
    data = json.loads(metrics.read_text(encoding="utf-8"))
    data["models"]["cnn"]["test"]["segment"]["f1_macro"] += 0.01
    metrics.write_text(json.dumps(data), encoding="utf-8")
    assert check.main(args(metrics, reference)) == 1
    assert "cnn segment" in capsys.readouterr().out


def test_a_last_digit_change_is_accepted(evaluated):
    metrics, reference = evaluated
    data = json.loads(metrics.read_text(encoding="utf-8"))
    data["models"]["cnn"]["test"]["segment"]["f1_macro"] += 0.001
    metrics.write_text(json.dumps(data), encoding="utf-8")
    assert check.main(args(metrics, reference)) == 0


def test_other_weights_are_reported(evaluated, capsys):
    metrics, reference = evaluated
    data = json.loads(metrics.read_text(encoding="utf-8"))
    data["models"]["baseline"]["weights_sha256"] = "0" * 64
    metrics.write_text(json.dumps(data), encoding="utf-8")
    assert check.main(args(metrics, reference)) == 1
    assert "different weights" in capsys.readouterr().out


def test_missing_metrics(tmp_path, capsys):
    assert check.main(args(tmp_path / "none.json", tmp_path / "ref.json")) == 2
