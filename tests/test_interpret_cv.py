"""Round 3 interpretability tests (F5-T07) on toy data, on CPU."""

import json

import numpy as np
import pytest
import torch
from test_cv import DEMON_BINS, FRAMES, LOFAR_BINS, engine, write_round3_toy

from sonar_pipeline.__main__ import main
from sonar_pipeline.interpret import bands
from sonar_pipeline.interpret_cv import (
    attention_by_band,
    class_token_attention,
    run_round3_interpretability,
    token_centres_hz,
)
from sonar_pipeline.transformer import SonarTransformer

MODELS = ("baseline", "cnn", "hybrid_aug", "transformer")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("round3")
    e = engine(tmp, seeds=(0, 1))
    summary = e.run(MODELS, tmp / "round3.json")
    return tmp, e, summary


def test_token_centres():
    hz = np.arange(1001) * 2.0
    transformer = token_centres_hz("transformer", 50, hz)
    assert transformer[0] == pytest.approx(19.0) and transformer[49] == pytest.approx(
        1979.0  # bins 980-999; the 2000 Hz bin is dropped
    )
    hybrid = token_centres_hz("hybrid_aug", 62, hz)
    assert hybrid[0] == pytest.approx(15.0) and hybrid[1] == pytest.approx(47.0)


def test_attention_is_summed_into_bands():
    band_list = bands(np.arange(101) * 2.0, 40.0)  # 0-40 ... 160-200
    values = np.array([0.1, 0.2, 0.3, 0.4])
    out = attention_by_band(values, np.array([10.0, 30.0, 50.0, 199.0]), band_list)
    assert out.tolist() == pytest.approx([0.3, 0.3, 0.0, 0.0, 0.4])


def test_class_token_attention_sums_to_one():
    model = SonarTransformer(
        lofar_frames=FRAMES, lofar_bins=LOFAR_BINS, demon_bins=DEMON_BINS, layers=2
    )
    attention = class_token_attention(
        model, torch.randn(10, FRAMES, LOFAR_BINS), torch.randn(10, DEMON_BINS), batch=4
    )
    assert attention.shape == (2, 2 + LOFAR_BINS // 20)
    assert attention.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert all(block.attention.keep_attention is False for block in model.blocks)


def test_interpretability_of_every_model_on_val_only(trained, monkeypatch):
    tmp, e, summary = trained

    def forbidden(*args, **kwargs):
        raise AssertionError("test used in the interpretability")

    monkeypatch.setattr(e, "test_tensors", forbidden)
    monkeypatch.setattr(e, "predict_test", forbidden)
    report = run_round3_interpretability(
        e, summary, tmp / "interp.json", log=lambda line: None
    )
    assert set(report["models"]) == set(MODELS)
    n_bands = len(report["lofar_bands"])
    base = report["models"]["baseline"]
    assert base["seeds"] == [0]
    assert len(base["measures"]["lofar"]["mean"]) == n_bands
    cnn = report["models"]["cnn"]["measures"]
    assert len(cnn["saliency_lofar"]["mean"]) == 4  # per class
    assert np.array(cnn["lofar_per_class"]["mean"]).shape == (n_bands, 4)
    for name in ("hybrid_aug", "transformer"):
        entry = report["models"][name]
        assert (
            np.array(entry["measures"]["attention_bands"]["mean"]).shape[1] == n_bands
        )
        assert entry["top_attention_bands"]
    assert "transformer_attention" in report["era_check"]["models"]
    # The toy's class lines (bins 10, 30, 50, 70 -> 20-160 Hz) matter more than the top band.
    lofar = np.array(cnn["lofar"]["mean"])
    assert lofar[:4].max() > lofar[4]
    saved = json.loads((tmp / "interp.json").read_text(encoding="utf-8"))
    assert saved["split"].startswith("val")


def test_cli_round3_interpretability(tmp_path, capsys):
    folder = write_round3_toy(tmp_path / "p")
    argv = ["train", "--round", "3", "--only", "baseline", "cnn", "--max-epochs", "1"]
    argv += ["--processed-dir", str(folder), "--splits", str(folder / "cv_splits.csv")]
    argv += ["--models-dir", str(tmp_path / "m"), "--outputs-dir", str(tmp_path / "o")]
    argv += ["--summary", str(tmp_path / "r" / "round3.json"), "--device", "cpu"]
    argv += ["--bootstrap", "5"]
    assert main(argv) == 0
    argv = ["evaluate", "--round", "3", "--interpretability", "--device", "cpu"]
    argv += ["--processed-dir", str(folder), "--splits", str(folder / "cv_splits.csv")]
    argv += ["--models-dir", str(tmp_path / "m"), "--results-dir", str(tmp_path / "r")]
    assert main(argv) == 0
    assert "top LOFAR bands" in capsys.readouterr().out
    assert (tmp_path / "r" / "round3_interpretability.json").is_file()


def test_cli_round3_evaluate_needs_interpretability():
    assert main(["evaluate", "--round", "3"]) == 1
