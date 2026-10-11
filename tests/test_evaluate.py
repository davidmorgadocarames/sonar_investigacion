"""evaluate tests (F3-T09) with toy features and toy-trained models. No real data here."""

import csv
import hashlib
import json

import numpy as np
import pytest

from sonar_pipeline.__main__ import main
from sonar_pipeline.baseline import train_baseline
from sonar_pipeline.cnn import train_cnn, train_cnn_aug
from sonar_pipeline.dataset import CLASSES

FIELDS = ["segment_id", "wav_path", "vessel_class", "vessel_group", "split"]
SPLIT_OF_VESSEL = ["train", "train", "train", "val", "val", "test"]


def write_toy(folder, seed=0):
    """4 classes x 6 vessels (3 train, 2 val, 1 test) x 4 segments, with a class-specific
    LOFAR line and DEMON peak."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    index, lofar, demon = [], [], []
    for k, cls in enumerate(CLASSES):
        for v, split in enumerate(SPLIT_OF_VESSEL):
            for _ in range(4):
                index.append((len(index), f"{cls}/{v}.wav", cls, f"{cls}/V{v}", split))
                frame = rng.normal(size=(4, 64))
                frame[:, 8 + 12 * k] += 6.0
                spectrum = rng.random(32) * 0.3
                spectrum[4 + 6 * k] = 1.0
                lofar.append(frame)
                demon.append(spectrum)
    with (folder / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows(index)
    for split in ("train", "val", "test"):
        rows = [r[0] for r in index if r[4] == split]
        np.savez(
            folder / f"features_{split}.npz",
            segment_id=np.array(rows),
            lofar=np.array(lofar, np.float32)[rows],
            demon=np.array(demon, np.float32)[rows],
            lofar_hz=np.arange(64.0),
            demon_hz=np.arange(32.0),
        )
    (folder / "features_meta.json").write_text(
        json.dumps({"split_sha256": "toy", "code_sha256": "toy"}), encoding="utf-8"
    )
    return folder


@pytest.fixture
def trained(tmp_path):
    folder = write_toy(tmp_path / "p")
    models = tmp_path / "m"
    train_baseline(folder, models, n_bootstrap=10)
    cnn_options = {
        "device_name": "cpu",
        "n_bootstrap": 10,
        "max_epochs": 3,
        "seeds": (0,),
    }
    train_cnn(folder, models, lr_grid=(1e-3,), weight_decay_grid=(1e-3,), **cnn_options)
    train_cnn_aug(folder, models, **cnn_options)
    return folder, models


def run(folder, models, tmp_path):
    argv = ["evaluate", "--processed-dir", str(folder), "--models-dir", str(models)]
    argv += ["--results-dir", str(tmp_path / "r"), "--figures-dir", str(tmp_path / "f")]
    return main([*argv, "--bootstrap", "20"])


def test_evaluate_scores_test_for_the_three_models(trained, tmp_path, capsys):
    folder, models = trained
    assert run(folder, models, tmp_path) == 0
    metrics = json.loads((tmp_path / "r" / "metrics.json").read_text("utf-8"))
    assert metrics["schema"] == 1
    assert set(metrics["models"]) == {"baseline", "cnn", "cnn_aug"}
    for entry in metrics["models"].values():
        assert entry["test"]["segment"]["n"] == 16  # 4 classes x 1 vessel x 4 segments
        assert entry["test"]["recording"]["n"] == 4
        assert set(entry["test"]["within_era"]) == {"2016", "2017"}
        assert entry["val"]["segment"]["n"] == 32  # copied from the model's JSON
        assert entry["inference"]["test"]["device"] == "cpu"
        weights = models / entry["weights"]
        assert (
            entry["weights_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
        )
    assert "grouped_cv" in metrics["models"]["baseline"]
    assert "comparison" in metrics["models"]["cnn_aug"]
    assert (
        metrics["dt005"] is None and metrics["permutation_test"] is None
    )  # not run here
    out = capsys.readouterr().out
    assert "scored once with the frozen" in out
    names = sorted(p.name for p in (tmp_path / "f").iterdir())
    assert names == [
        "results_confusion.png",
        "results_f1.png",
        "results_roc_recording.png",
        "results_roc_segment.png",
    ]


def test_evaluate_includes_dt005_and_permutation_results(trained, tmp_path):
    folder, models = trained
    results = tmp_path / "r"
    results.mkdir()
    (results / "dt005_duration.json").write_text('{"val": 1}', encoding="utf-8")
    (results / "permutation_test.json").write_text(
        '{"p_value": 0.04}', encoding="utf-8"
    )
    assert run(folder, models, tmp_path) == 0
    metrics = json.loads((results / "metrics.json").read_text("utf-8"))
    assert metrics["dt005"] == {"val": 1}
    assert metrics["permutation_test"] == {"p_value": 0.04}


def test_evaluate_needs_every_model(tmp_path):
    folder = write_toy(tmp_path / "p")
    with pytest.raises(FileNotFoundError, match="baseline.npz"):
        run(folder, tmp_path / "m", tmp_path)


def test_plot_only_redraws_from_metrics_json(trained, tmp_path, monkeypatch):
    folder, models = trained
    assert run(folder, models, tmp_path) == 0
    for png in (tmp_path / "f").iterdir():
        png.unlink()

    def no_scoring(*args, **kwargs):
        raise AssertionError("plot-only must not score test")

    monkeypatch.setattr("sonar_pipeline.evaluate.evaluate_split", no_scoring)
    argv = ["evaluate", "--plot-only", "--results-dir", str(tmp_path / "r")]
    assert main([*argv, "--figures-dir", str(tmp_path / "f")]) == 0
    assert len(list((tmp_path / "f").iterdir())) == 4


def test_metrics_record_training_hardware_and_code(trained, tmp_path):
    folder, models = trained
    assert run(folder, models, tmp_path) == 0
    metrics = json.loads((tmp_path / "r" / "metrics.json").read_text("utf-8"))
    assert len(metrics["code_sha256"]) == 64
    for entry in metrics["models"].values():
        assert entry["training_hardware"]["device"] == "cpu"
        assert len(entry["training_code_sha256"]) == 64
