"""CNN tests (F3-T07) on toy data, on CPU. No real training here."""

import csv
import json

import numpy as np
import pytest
import torch

from sonar_pipeline.__main__ import main
from sonar_pipeline.cnn import (
    SonarCNN,
    load_cnn,
    pool_to,
    predict_proba,
    select_device,
    split_parameters,
    train_cnn,
    train_run,
)
from sonar_pipeline.dataset import CLASSES, class_weights, load_split
from sonar_pipeline.metrics import torch_cost

FIELDS = ["segment_id", "wav_path", "vessel_class", "vessel_group", "split"]
FRAMES, LOFAR_BINS, DEMON_BINS = 4, 64, 32


def write_toy(folder, seed=0):
    """4 classes x 6 vessels (4 train, 2 val) x 4 segments. Class k has a tonal line at LOFAR
    bin 8 + 12 k and a DEMON peak at bin 4 + 6 k, over noise."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    index, lofar, demon = [], [], []
    for k, cls in enumerate(CLASSES):
        for v in range(6):
            split = "train" if v < 4 else "val"
            for _ in range(4):
                index.append((len(index), f"{cls}/{v}.wav", cls, f"{cls}/V{v}", split))
                frame = rng.normal(size=(FRAMES, LOFAR_BINS))
                frame[:, 8 + 12 * k] += 6.0
                spectrum = rng.random(DEMON_BINS) * 0.3
                spectrum[4 + 6 * k] = 1.0
                lofar.append(frame)
                demon.append(spectrum)
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
            lofar_hz=np.arange(float(LOFAR_BINS)),
            demon_hz=np.arange(float(DEMON_BINS)),
        )
    (folder / "features_meta.json").write_text(
        json.dumps({"split_sha256": "toy", "code_sha256": "toy"}), encoding="utf-8"
    )
    return folder


def test_forward_shape_and_size_on_real_input_shapes():
    model = SonarCNN().eval()
    logits = model(torch.zeros(3, 11, 1001), torch.zeros(3, 301))
    assert logits.shape == (3, len(CLASSES))
    cost = torch_cost(model, (torch.zeros(1, 11, 1001), torch.zeros(1, 301)))
    assert 70_000 < cost["parameters"] < 90_000  # D-018: about 80 000
    assert cost["flops_per_segment"] > 0


def test_pool_to_averages_equal_groups():
    x = torch.arange(12.0).reshape(1, 1, 12)
    assert pool_to(x, 4).flatten().tolist() == [1.0, 4.0, 7.0, 10.0]
    assert pool_to(torch.ones(1, 1, 31), 8).shape == (1, 1, 8)  # padded up to 32


def test_weight_decay_skips_biases_and_batchnorm():
    model = SonarCNN()
    decay, no_decay = split_parameters(model)
    assert all(p.ndim > 1 for p in decay["params"])
    assert all(p.ndim == 1 for p in no_decay["params"])
    assert no_decay["weight_decay"] == 0.0
    n = sum(p.numel() for p in model.parameters())
    assert sum(p.numel() for g in (decay, no_decay) for p in g["params"]) == n


def test_normalisation_is_saved_with_the_weights(tmp_path):
    model = SonarCNN()
    model.set_normalisation(np.full((2, 3), 5.0), np.full(4, 0.5) + np.arange(4))
    path = tmp_path / "w.pt"
    torch.save(model.state_dict(), path)
    loaded = load_cnn(path)
    assert float(loaded.lofar_mean) == 5.0
    assert float(loaded.demon_std) == pytest.approx(np.std(np.arange(4)))
    assert not loaded.training  # evaluation mode (BatchNorm and dropout frozen)


def test_cuda_request_without_gpu_fails():
    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA")
    assert select_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA requested"):
        select_device("cuda")


def toy_inputs(folder):
    train, val = load_split(folder, "train"), load_split(folder, "val")
    tensors = {
        "train_lofar": torch.as_tensor(train.lofar),
        "train_demon": torch.as_tensor(train.demon),
        "val_lofar": torch.as_tensor(val.lofar),
        "val_demon": torch.as_tensor(val.demon),
    }
    common = {
        "tensors": tensors,
        "labels": {"train": train.labels, "val": val.labels},
        "weights": torch.as_tensor(class_weights(train.labels), dtype=torch.float32),
        "device": torch.device("cpu"),
        "normalisation": (train.lofar, train.demon),
    }
    return common


def test_same_seed_gives_the_same_run(tmp_path):
    common = toy_inputs(write_toy(tmp_path / "p"))
    a = train_run(lr=1e-3, weight_decay=1e-2, seed=3, max_epochs=4, **common)
    b = train_run(lr=1e-3, weight_decay=1e-2, seed=3, max_epochs=4, **common)
    assert a.history == b.history
    for key in a.state:
        assert torch.equal(a.state[key], b.state[key])


def test_early_stopping_stops_after_patience(tmp_path):
    common = toy_inputs(write_toy(tmp_path / "p"))
    run = train_run(
        lr=1e-3, weight_decay=0.0, seed=0, max_epochs=50, patience=3, **common
    )
    # The toy is learnt fast; once val F1 stops improving, 3 more epochs and it stops.
    assert run.epochs_run <= run.best_epoch + 3
    assert run.epochs_run < 50


def test_train_cnn_learns_the_toy_and_reloads(tmp_path):
    folder = write_toy(tmp_path / "p")
    report = train_cnn(
        folder,
        tmp_path / "m",
        device_name="cpu",
        n_bootstrap=20,
        max_epochs=15,
        patience=5,
        lr_grid=(1e-3,),
        weight_decay_grid=(1e-3, 1e-2),
        seeds=(0, 1),
    )
    assert report["val"]["segment"]["f1_macro"] >= 0.9
    assert (
        len(report["training"]["search"]) == 2 and len(report["training"]["seeds"]) == 2
    )
    assert report["training"]["device"] == "cpu"
    assert report["cost"]["parameters"] > 0
    assert "test" not in report
    saved = json.loads((tmp_path / "m" / "cnn.json").read_text("utf-8"))
    assert saved["config"]["seed"] in (0, 1)
    # The saved weights give back the reported val scores.
    model = load_cnn(tmp_path / "m" / "cnn.pt")
    val = load_split(folder, "val")
    probs = predict_proba(model, torch.as_tensor(val.lofar), torch.as_tensor(val.demon))
    f1 = report["val"]["segment"]["f1_macro"]
    from sonar_pipeline.metrics import f1_scores

    assert f1_scores(val.labels, probs.argmax(axis=1))[0] == pytest.approx(f1)


def test_cli_trains_the_cnn(tmp_path, capsys):
    folder = write_toy(tmp_path / "p")
    argv = ["train", "--model", "cnn", "--processed-dir", str(folder)]
    argv += ["--models-dir", str(tmp_path / "m"), "--device", "cpu"]
    assert main([*argv, "--max-epochs", "2", "--bootstrap", "20"]) == 0
    out = capsys.readouterr().out
    assert "CNN on cpu: 4 configurations" in out and "test not used" in out
    assert (tmp_path / "m" / "cnn.pt").is_file()
