"""Interpretability tests (F3-T10) on toy data whose important bands are known."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_cnn import write_toy  # train and val only: no test file at all

from sonar_pipeline.__main__ import main
from sonar_pipeline.baseline import train_baseline
from sonar_pipeline.cnn import train_cnn, train_cnn_aug
from sonar_pipeline.dataset import CLASSES
from sonar_pipeline.interpret import bands, era_check, run_interpretability

# In the toy, class k has a LOFAR line at 8 + 12 k Hz (1 Hz bins) and a DEMON peak at bin 4 + 6 k.
LINE_HZ = [8 + 12 * k for k in range(4)]


@pytest.fixture
def trained(tmp_path):
    folder = write_toy(tmp_path / "p")
    models = tmp_path / "m"
    train_baseline(folder, models, n_bootstrap=10)
    options = {"device_name": "cpu", "n_bootstrap": 10, "max_epochs": 8, "seeds": (0,)}
    train_cnn(folder, models, lr_grid=(1e-3,), weight_decay_grid=(1e-3,), **options)
    train_cnn_aug(folder, models, **options)
    return folder, models


def test_bands_cover_every_bin_once():
    hz = np.arange(1001) * 2.0  # LOFAR axis: 0-2000 Hz
    band_list = bands(hz, 40.0)
    assert len(band_list) == 50
    covered = np.concatenate([b["bins"] for b in band_list])
    assert sorted(covered.tolist()) == list(
        range(1001)
    )  # the 2000 Hz bin goes to the last
    assert band_list[-1]["hi"] == 2000.0


def test_important_bands_are_the_class_lines(trained, tmp_path):
    folder, models = trained
    report = run_interpretability(
        folder,
        models,
        tmp_path / "r",
        tmp_path / "f",
        lofar_band_hz=4.0,
        demon_band_hz=3.0,
    )
    line_bands = {int(h // 4) * 4.0 for h in LINE_HZ}
    top_baseline = sorted(report["baseline"]["lofar"], key=lambda b: -b["value"])[:4]
    assert {b["lo_hz"] for b in top_baseline} <= line_bands
    for name in ("cnn", "cnn_aug"):
        top_cnn = sorted(
            report["cnn"][name]["occlusion_lofar"], key=lambda b: -b["value"]
        )[:2]
        assert {b["lo_hz"] for b in top_cnn} <= line_bands, name
    maps = np.load(tmp_path / "r" / "interpretability_maps.npz")
    assert maps["cnn_saliency_lofar"].shape == (len(CLASSES), 4, 64)
    assert (tmp_path / "f" / "interpretability_bands.png").is_file()
    assert report["split"] == "val"


def test_metrics_json_gets_the_summary_and_keeps_the_rest(trained, tmp_path):
    folder, models = trained
    results = tmp_path / "r"
    results.mkdir()
    (results / "metrics.json").write_text(
        json.dumps({"schema": 1, "models": {"x": 1}}), encoding="utf-8"
    )
    run_interpretability(folder, models, results, lofar_band_hz=4.0, demon_band_hz=3.0)
    metrics = json.loads((results / "metrics.json").read_text("utf-8"))
    assert metrics["models"] == {"x": 1} and metrics["schema"] == 1
    assert "era_check" in metrics["interpretability"]


def fake_split(spectra, classes, groups, hz):
    """One segment per row; lofar with a single frame."""
    return SimpleNamespace(
        lofar=np.array(spectra)[:, None, :],
        labels=np.array([CLASSES.index(c) for c in classes]),
        vessel_group=np.array(groups),
        lofar_hz=hz,
    )


def test_era_check_finds_a_line_shared_by_one_era():
    hz = np.arange(20.0)
    spectra, classes, groups = [], [], []
    for v, cls in enumerate(
        ["Cargo", "Tug", "Cargo", "Tug", "Passengership", "Tanker"] * 2
    ):
        s = np.zeros(20)
        if cls in ("Cargo", "Tug"):  # 2017: every vessel has a line at 12 Hz
            s[12] = 6.0
        spectra.append(s)
        classes.append(cls)
        groups.append(f"{cls}/V{v}")
    split = fake_split(spectra, classes, groups, hz)
    empty = fake_split(np.zeros((0, 20)), [], [], hz)
    band_list = bands(hz, 4.0)
    importance = np.zeros(len(band_list))
    importance[3] = 1.0  # the model relies on the 12-16 Hz band
    result = era_check(split, empty, band_list, {"toy": importance})
    assert result["era_lines_hz"] == [12.0]
    assert 12.0 in result["top_difference_bands_hz"]
    assert result["models"]["toy"]["top_overlap"] >= 1


def test_cli_flag(trained, tmp_path, capsys):
    folder, models = trained
    argv = ["evaluate", "--interpretability", "--processed-dir", str(folder)]
    argv += ["--models-dir", str(models), "--results-dir", str(tmp_path / "r")]
    assert main([*argv, "--figures-dir", str(tmp_path / "f")]) == 0
    out = capsys.readouterr().out
    assert "test not used" in out and "era lines" in out
