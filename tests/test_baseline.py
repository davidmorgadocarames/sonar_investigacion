"""Baseline tests (F3-T06) on toy features whose answer is known. No real training here."""

import csv
import json

import numpy as np
import pytest

from sonar_pipeline.__main__ import main
from sonar_pipeline.baseline import (
    C_GRID,
    load_baseline,
    make_model,
    predict_proba,
    shuffle_by_group,
)
from sonar_pipeline.dataset import CLASSES, baseline_features, load_split

FIELDS = ["segment_id", "wav_path", "vessel_class", "vessel_group", "split"]


def write_toy(folder, separable=True, seed=0):
    """4 classes x 6 vessels (4 train, 2 val) x 3 segments; no test file at all.

    Separable: class k raises LOFAR bin k by 3 (noise std 1); otherwise only noise.
    """
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    index, lofar, demon = [], [], []
    for k, cls in enumerate(CLASSES):
        for v in range(6):
            split = "train" if v < 4 else "val"
            for _ in range(3):
                index.append((len(index), f"{cls}/{v}.wav", cls, f"{cls}/V{v}", split))
                frame = rng.normal(size=(2, 5))
                if separable:
                    frame[:, k] += 3.0
                lofar.append(frame)
                demon.append(rng.normal(size=3))
    with (folder / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows(index)
    for split in ("train", "val"):
        rows = [r[0] for r in index if r[4] == split]
        np.savez(
            folder / f"features_{split}.npz",
            segment_id=np.array(rows),
            lofar=np.array(lofar, np.float32)[rows],
            demon=np.array(demon, np.float32)[rows],
            lofar_hz=np.arange(5.0),
            demon_hz=np.arange(3.0),
        )
    (folder / "features_meta.json").write_text(
        json.dumps({"split_sha256": "toy", "code_sha256": "toy"}), encoding="utf-8"
    )
    return folder


def train(folder, models, *extra):
    argv = ["train", "--model", "baseline", "--processed-dir", str(folder)]
    return main([*argv, "--models-dir", str(models), "--bootstrap", "50", *extra])


def test_baseline_learns_a_separable_toy(tmp_path, capsys):
    folder = write_toy(tmp_path / "p")
    assert train(folder, tmp_path / "m") == 0
    report = json.loads((tmp_path / "m" / "baseline.json").read_text("utf-8"))
    assert report["val"]["segment"]["f1_macro"] == 1.0
    assert report["config"]["c"] in C_GRID
    assert [row["c"] for row in report["training"]["search"]] == list(C_GRID)
    assert report["config"]["n_features"] == 5 + 3  # time-mean LOFAR bins + DEMON bins
    assert report["cost"]["parameters"] == 8 * 4 + 4
    assert len(report["grouped_cv"]["f1_macro"]) == 5
    assert report["inference"]["segments"] == 24
    assert "test" not in report
    assert "test not used" in capsys.readouterr().out


def test_saved_weights_reproduce_scikit_learn(tmp_path):
    folder = write_toy(tmp_path / "p")
    assert train(folder, tmp_path / "m") == 0
    weights = load_baseline(tmp_path / "m" / "baseline.npz")
    trainset, val = load_split(folder, "train"), load_split(folder, "val")
    reference = make_model(float(weights["c"])).fit(
        baseline_features(trainset), trainset.labels
    )
    x_val = baseline_features(val)
    np.testing.assert_allclose(
        predict_proba(weights, x_val), reference.predict_proba(x_val), atol=1e-10
    )


def test_shuffle_by_group_keeps_groups_whole():
    y = np.repeat([0, 0, 1, 1, 2, 3], 3)
    groups = np.repeat(["a", "b", "c", "d", "e", "f"], 3)
    rng = np.random.default_rng(0)
    for _ in range(20):
        shuffled = shuffle_by_group(y, groups, rng)
        for g in "abcdef":
            assert len(set(shuffled[groups == g])) == 1  # one label per vessel
        assert sorted(shuffled[::3]) == sorted(y[::3])  # same groups per class


def test_permutation_test_on_a_separable_toy(tmp_path, capsys):
    folder = write_toy(tmp_path / "p")
    assert train(folder, tmp_path / "m") == 0
    results = tmp_path / "r"
    assert (
        train(
            folder,
            tmp_path / "m",
            "--permutation-test",
            "19",
            "--results-dir",
            str(results),
        )
        == 0
    )
    result = json.loads((results / "permutation_test.json").read_text("utf-8"))
    assert result["observed_f1_macro"] == 1.0
    assert len(result["null_f1_macro"]) == 19
    assert result["null_mean"] < 0.7
    assert result["p_value"] == pytest.approx(1 / 20)  # no permutation reaches 1.0
    assert "p = 0.0500" in capsys.readouterr().out


def test_permutation_test_needs_a_trained_model(tmp_path):
    folder = write_toy(tmp_path / "p")
    with pytest.raises(FileNotFoundError):
        train(folder, tmp_path / "m", "--permutation-test", "3")


def test_permutation_test_only_for_baseline(tmp_path, capsys):
    assert main(["train", "--model", "cnn", "--permutation-test", "3"]) == 1
    assert "only for --model baseline" in capsys.readouterr().err


def test_cv_folds_never_share_a_vessel():
    from sonar_pipeline.baseline import CV_FOLDS, cv_folds

    y = np.repeat([0, 1, 2, 3], 30)
    groups = np.array(
        [f"{k}-{i // 3}" for i, k in enumerate(y)]
    )  # 10 vessels per class
    folds = cv_folds(y, groups)
    assert len(folds) == CV_FOLDS
    for train_rows, held_out in folds:
        assert not set(groups[train_rows]) & set(groups[held_out])
    held = np.concatenate([h for _, h in folds])
    assert sorted(held.tolist()) == list(
        range(len(y))
    )  # every row held out exactly once


def test_select_c_matches_the_trained_choice(tmp_path):
    from sonar_pipeline.baseline import select_c

    folder = write_toy(tmp_path / "p")
    assert train(folder, tmp_path / "m") == 0
    report = json.loads((tmp_path / "m" / "baseline.json").read_text("utf-8"))
    trainset, val = load_split(folder, "train"), load_split(folder, "val")
    chosen = select_c(
        baseline_features(trainset), trainset.labels, baseline_features(val), val.labels
    )
    assert chosen == report["config"]["c"]
    assert report["code_sha256"] and len(report["code_sha256"]) == 64


def test_permutation_test_chooses_c_inside_every_permutation(tmp_path):
    folder = write_toy(tmp_path / "p")
    assert train(folder, tmp_path / "m") == 0
    results = tmp_path / "r"
    extra = ["--permutation-test", "4", "--results-dir", str(results)]
    assert train(folder, tmp_path / "m", *extra) == 0
    result = json.loads((results / "permutation_test.json").read_text("utf-8"))
    assert len(result["null_c"]) == 4 and set(result["null_c"]) <= set(C_GRID)
    assert result["c"] == result["saved_c"]
