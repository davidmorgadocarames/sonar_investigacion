"""DT-005 check tests (F3-T05) on toy features whose answer is known."""

import csv
import json

import numpy as np
import pytest

from sonar_pipeline.__main__ import main
from sonar_pipeline.dataset import CLASSES

FIELDS = ["segment_id", "wav_path", "vessel_class", "vessel_group", "split"]


def write_toy(folder, durations):
    """2 segments per recording; 3 train and 2 val recordings per class; no test file at all.

    durations(k, i) gives the duration of recording i of class k.
    """
    folder.mkdir(parents=True)
    index, vessels = [], []
    for split, n_rec in (("train", 3), ("val", 2)):
        for k, cls in enumerate(CLASSES):
            for i in range(n_rec):
                path = f"{cls}/{split}{i}.wav"
                vessels.append((path, durations(k, i)))
                for _ in range(2):
                    index.append((len(index), path, cls, f"{cls}/{split}{i}", split))
    with (folder / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows(index)
    for split in ("train", "val"):
        ids = [r[0] for r in index if r[4] == split]
        np.savez(
            folder / f"features_{split}.npz",
            segment_id=np.array(ids),
            lofar=np.zeros((len(ids), 2, 3), np.float32),
            demon=np.zeros((len(ids), 4), np.float32),
            lofar_hz=np.arange(3.0),
            demon_hz=np.arange(4.0),
        )
    vessels_csv = folder / "vessels.csv"
    with vessels_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["wav_path", "wav_duration_s"])
        writer.writerows(vessels)
    return folder, vessels_csv


def run(folder, vessels_csv, results):
    return main(
        [
            "train",
            "--model",
            "duration",
            "--processed-dir",
            str(folder),
            "--vessels",
            str(vessels_csv),
            "--results-dir",
            str(results),
            "--bootstrap",
            "50",
        ]
    )


def test_duration_that_tells_the_class_scores_high(tmp_path, capsys):
    # Each class has its own duration range: duration alone separates them perfectly.
    folder, vessels = write_toy(tmp_path / "p", lambda k, i: 50.0 * (k + 1) + i)
    assert run(folder, vessels, tmp_path / "r") == 0
    result = json.loads((tmp_path / "r" / "dt005_duration.json").read_text("utf-8"))
    assert result["val"]["segment"]["f1_macro"] == 1.0
    assert result["val"]["recording"]["n"] == 8
    assert "test" not in result  # test is never loaded (there is no test file here)
    assert "scored on val (test not used)" in capsys.readouterr().out


def test_uninformative_duration_scores_low(tmp_path):
    # The same duration for every recording: nothing to learn.
    folder, vessels = write_toy(tmp_path / "p", lambda k, i: 100.0)
    assert run(folder, vessels, tmp_path / "r") == 0
    result = json.loads((tmp_path / "r" / "dt005_duration.json").read_text("utf-8"))
    assert result["val"]["segment"]["f1_macro"] < 0.5


def test_missing_duration_fails(tmp_path):
    folder, vessels = write_toy(tmp_path / "p", lambda k, i: 10.0)
    lines = vessels.read_text("utf-8").splitlines()
    vessels.write_text("\n".join(lines[:-1]) + "\n", "utf-8")
    with pytest.raises(ValueError, match="no duration"):
        run(folder, vessels, tmp_path / "r")


@pytest.mark.parametrize("model", ["cnn-aug"])
def test_models_not_yet_implemented_fail(model, tmp_path, capsys):
    assert main(["train", "--model", model, "--results-dir", str(tmp_path)]) == 1
    assert "not implemented yet" in capsys.readouterr().err
