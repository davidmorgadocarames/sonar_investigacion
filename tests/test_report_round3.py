"""Round 3 report tests (F5-T08): details, the 15 figures and the report chapter, on toy data."""

import json

import numpy as np
import pytest
from test_cv import TOY_CONFIGS, engine, write_round3_toy

from sonar_pipeline.__main__ import main
from sonar_pipeline.cv import MODELS
from sonar_pipeline.figures_round3 import FIGURES, corrected_interval, plot_round3
from sonar_pipeline.interpret_cv import run_round3_interpretability
from sonar_pipeline.report_round3 import retrain_lines, round3_section, summary_lines
from sonar_pipeline.round3_details import collect_details, curves, variance


@pytest.fixture(scope="module")
def round3(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("round3")
    e = engine(tmp, seeds=(0, 1))
    summary = e.run(MODELS, tmp / "round3.json")
    details = collect_details(e, summary, tmp / "details.json", log=lambda line: None)
    interp = run_round3_interpretability(
        e, summary, tmp / "interp.json", log=lambda line: None
    )
    return tmp, e, summary, details, interp


def test_details_of_every_model(round3):
    _, e, summary, details, _ = round3
    assert list(details["models"]) == list(MODELS)
    cnn = details["models"]["cnn"]
    assert len(cnn["generalisation"]) == len(e.partitions)
    assert all(0 <= g["train_f1_macro"] <= 1 for g in cnn["generalisation"])
    assert [g["test_f1_macro"] for g in cnn["generalisation"]] == [
        p["test_f1_macro_segment"] for p in summary["models"]["cnn"]["per_partition"]
    ]
    assert cnn["speed"]["trainings"] == 2 * len(e.partitions)  # seeds 0 and 1
    assert cnn["speed"]["weights_bytes"] > 0
    assert details["models"]["baseline"]["curves"] is None
    assert cnn["curves"]["runs"][0] == 2 * len(e.partitions)
    assert set(cnn["roc"]) == {"Cargo", "Passengership", "Tanker", "Tug"}
    assert 0 <= cnn["calibration"]["brier"] <= 2
    assert sum(v["tests"] for v in cnn["vessels"].values()) == 4 * len(e.partitions)
    pairs = details["agreement_recording_level"]
    assert len(pairs) == len(MODELS) * (len(MODELS) - 1) // 2
    assert all(0 <= p["same_prediction"] <= 1 for p in pairs.values())


def test_variance_splits_partition_and_seed():
    rows = [
        {"test_f1_macro_per_seed": [0.2, 0.4]},
        {"test_f1_macro_per_seed": [0.6, 0.8]},
    ]
    v = variance(rows)
    assert v["partition_std"] == pytest.approx(np.std([0.3, 0.7], ddof=1))
    assert v["seed_std"] == pytest.approx(np.std([0.2, 0.4], ddof=1))


def test_curves_average_the_runs_still_going():
    records = [
        {
            "training": {
                "best_epoch": 1,
                "history": [
                    {
                        "train_loss": 1.0,
                        "val_loss": 2.0,
                        "val_f1_macro": 0.1,
                        "lr": 1e-3,
                    },
                    {
                        "train_loss": 0.5,
                        "val_loss": 2.0,
                        "val_f1_macro": 0.3,
                        "lr": 1e-3,
                    },
                ],
            }
        },
        {
            "training": {
                "best_epoch": 1,
                "history": [
                    {
                        "train_loss": 3.0,
                        "val_loss": 2.0,
                        "val_f1_macro": 0.3,
                        "lr": 1e-3,
                    },
                ],
            }
        },
    ]
    c = curves(records)
    assert c["runs"] == [2, 1]
    assert c["train_loss"]["mean"] == pytest.approx([2.0, 0.5])


def test_corrected_interval_contains_the_mean():
    c = {
        "partitions": 25,
        "test_train_ratio": 0.375,
        "std_difference": 0.1,
        "df": 24,
        "mean_difference": 0.05,
    }
    lo, hi = corrected_interval(c)
    assert lo < 0.05 < hi
    assert hi - 0.05 == pytest.approx(0.05 - lo)


def test_the_fifteen_figures(round3, tmp_path):
    _, _, summary, details, interp = round3
    paths = plot_round3(summary, details, interp, tmp_path / "figs")
    assert len(paths) == 15 == len(FIGURES)
    for path in paths.values():
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_report_chapter_reads_the_numbers(round3):
    _, _, summary, details, interp = round3
    text = "\n".join(round3_section(summary, details, interp, "round3"))
    cnn = summary["models"]["cnn"]["test"]["f1_macro_segment"]
    assert f"{cnn['mean']:.3f}".replace(".", ",") in text
    assert text.count("![Ronda 3") == 15
    for name in TOY_CONFIGS["transformer"]:
        assert f"{name} " in text
    lines = summary_lines(summary)
    assert "Ronda 3" in lines[0]


def test_cli_details(tmp_path, capsys):
    folder = write_round3_toy(tmp_path / "p")
    argv = ["train", "--round", "3", "--only", "baseline", "cnn", "--max-epochs", "1"]
    argv += ["--processed-dir", str(folder), "--splits", str(folder / "cv_splits.csv")]
    argv += ["--models-dir", str(tmp_path / "m"), "--outputs-dir", str(tmp_path / "o")]
    argv += ["--summary", str(tmp_path / "r" / "round3.json"), "--device", "cpu"]
    argv += ["--bootstrap", "5"]
    assert main(argv) == 0
    argv = ["evaluate", "--round", "3", "--details", "--device", "cpu"]
    argv += ["--processed-dir", str(folder), "--splits", str(folder / "cv_splits.csv")]
    argv += ["--models-dir", str(tmp_path / "m"), "--results-dir", str(tmp_path / "r")]
    argv += ["--outputs-dir", str(tmp_path / "o")]
    assert main(argv) == 0
    assert "train F1" in capsys.readouterr().out
    saved = json.loads(
        (tmp_path / "r" / "round3_details.json").read_text(encoding="utf-8")
    )
    assert set(saved["models"]) == {"baseline", "cnn"}


def test_details_include_gradient_norm_and_data_time(round3):
    _, _, _, details, _ = round3
    cnn = details["models"]["cnn"]
    assert cnn["curves"]["grad_norm_mean"]["mean"][0] > 0
    assert 0 <= cnn["speed"]["minibatch_share_of_training"] <= 1
    assert cnn["speed"]["features_from_disk_seconds"] > 0
    assert details["models"]["baseline"]["speed"]["minibatch_share_of_training"] is None


def test_report_shows_the_retraining_against_the_initial_results(round3):
    _, _, summary, details, interp = round3
    text = "\n".join(
        round3_section(summary, details, interp, "round3", initial=summary)
    )
    assert "Resultados reentrenados (F5-T10)" in text
    assert "docs/results/round3_initial/" in text
    rows = [row for row in retrain_lines(summary, summary) if row.startswith("| ")][1:]
    assert len(rows) == len(summary["models"])
    assert all(row.endswith("| 0,000 |") for row in rows)  # same results: no difference
    plain = "\n".join(round3_section(summary, details, interp, "round3"))
    assert "Resultados reentrenados" not in plain
