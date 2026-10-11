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


@pytest.mark.parametrize(
    "size", [3, 8, 18, 31]
)  # also fewer values than groups; 18 and
# 31 are the DEMON and LOFAR lengths before the head
def test_pool_to_has_no_dead_positions_on_real_shapes(size):
    # Review of F3 (M2): zero padding left DEMON positions 6 and 7 always at 0.
    pooled = pool_to(torch.ones(1, 1, size), 8)
    assert pooled.flatten().tolist() == [1.0] * 8
    x = torch.randn(1, 1, size, requires_grad=True)
    pool_to(x, 8).sum().backward()
    assert torch.all(x.grad > 0)  # every input reaches the output


def test_real_branch_lengths_before_pooling():
    model = SonarCNN().eval()
    lofar = model.lofar(torch.zeros(1, 1, 11, 1001)).mean(dim=2)
    demon = model.demon(torch.zeros(1, 1, 301))
    assert lofar.shape[-1] == 31 and demon.shape[-1] == 18


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


# --------------------------------------------------------------------------- augmentation


def batch(n=6):
    lofar = torch.arange(n * 11 * 1001, dtype=torch.float32).reshape(n, 11, 1001)
    demon = torch.arange(n * 301, dtype=torch.float32).reshape(n, 301) + 1.0
    return lofar, demon


def test_augment_never_applies_with_probability_zero():
    from sonar_pipeline.cnn import make_augment

    augment = make_augment(-1.0, -2.0, 2.0, 1 / 3, probability=0.0)
    lofar, demon = batch()
    a, b = augment(lofar, demon, torch.Generator().manual_seed(0))
    assert torch.equal(a, lofar) and torch.equal(b, demon)


def test_augment_masks_and_shifts_with_probability_one():
    from sonar_pipeline.cnn import make_augment

    augment = make_augment(-1.0, -2.0, 2.0, 1 / 3, probability=1.0)
    lofar, demon = batch()
    original = lofar.clone()
    a, b = augment(lofar, demon, torch.Generator().manual_seed(0))
    assert torch.equal(lofar, original)  # inputs untouched
    for i in range(len(a)):
        masked_bins = (a[i] == -1.0).all(dim=0)  # whole frequency columns
        assert (
            1 <= int(masked_bins.sum()) <= 2 * 50
        )  # 1-2 bands of at most 50 bins (100 Hz)
        masked_frames = (a[i] == -1.0).all(dim=1)
        assert 1 <= int(masked_frames.sum()) <= 2  # one block of 1-2 frames
        demon_masked = int((b[i] == -2.0).sum())
        assert 1 <= demon_masked <= 10  # one DEMON band of at most 10 bins (3.3 Hz)
        # The time shift is a circular roll: every original frame is still there.
        kept = a[i][~masked_frames][:, ~masked_bins]
        frames = {tuple(f.tolist()) for f in original[i][:, ~masked_bins]}
        assert all(tuple(f.tolist()) in frames for f in kept)
        assert not torch.equal(a[i][~masked_frames], original[i][~masked_frames])


def test_augment_is_reproducible_with_a_seed():
    from sonar_pipeline.cnn import make_augment

    augment = make_augment(0.0, 0.0, 2.0, 1 / 3)
    lofar, demon = batch()
    a1, b1 = augment(lofar, demon, torch.Generator().manual_seed(7))
    a2, b2 = augment(lofar, demon, torch.Generator().manual_seed(7))
    assert torch.equal(a1, a2) and torch.equal(b1, b2)


def test_cnn_aug_needs_the_plain_cnn_first(tmp_path):
    from sonar_pipeline.cnn import train_cnn_aug

    folder = write_toy(tmp_path / "p")
    with pytest.raises(FileNotFoundError):
        train_cnn_aug(folder, tmp_path / "m", device_name="cpu", n_bootstrap=10)


def test_cli_trains_cnn_aug_with_the_cnn_configuration(tmp_path, capsys):
    folder = write_toy(tmp_path / "p")
    base = ["--processed-dir", str(folder), "--models-dir", str(tmp_path / "m")]
    base += ["--device", "cpu", "--max-epochs", "2", "--bootstrap", "20"]
    assert main(["train", "--model", "cnn", *base]) == 0
    assert main(["train", "--model", "cnn-aug", *base]) == 0
    plain = json.loads((tmp_path / "m" / "cnn.json").read_text("utf-8"))
    aug = json.loads((tmp_path / "m" / "cnn_aug.json").read_text("utf-8"))
    assert (aug["config"]["lr"], aug["config"]["weight_decay"]) == (
        plain["config"]["lr"],
        plain["config"]["weight_decay"],
    )
    assert len(aug["training"]["search"]) == 1 and len(aug["training"]["seeds"]) == 3
    assert aug["config"]["augment"]["probability"] == 0.5
    assert len(aug["comparison"]["cnn_seed_val_f1"]) == 3
    assert "same seeds on val" in capsys.readouterr().out
    assert (tmp_path / "m" / "cnn_aug.pt").is_file()


# --------------------------------------------------------------------------- review of F3


def test_threads_are_fixed():
    from sonar_pipeline.cnn import TORCH_THREADS, set_determinism

    set_determinism(0)
    assert torch.get_num_threads() == TORCH_THREADS


def test_augmentation_has_its_own_generator(tmp_path):
    from sonar_pipeline.cnn import AUGMENT_SEED_OFFSET

    common = toy_inputs(write_toy(tmp_path / "p"))
    seen = []

    def spy(lofar, demon, generator):
        seen.append(generator.initial_seed())
        return lofar, demon

    train_run(lr=1e-3, weight_decay=0.0, seed=4, max_epochs=1, augment=spy, **common)
    assert set(seen) == {
        4 + AUGMENT_SEED_OFFSET
    }  # not the shuffling generator (seed 4)


def test_same_minibatch_order_with_and_without_augmentation(tmp_path):
    # With separate generators, an identity augmentation must not change the run.
    common = toy_inputs(write_toy(tmp_path / "p"))

    def identity(lofar, demon, generator):
        torch.rand(5, generator=generator)  # consumes random numbers, changes nothing
        return lofar, demon

    plain = train_run(lr=1e-3, weight_decay=0.0, seed=2, max_epochs=3, **common)
    augmented = train_run(
        lr=1e-3, weight_decay=0.0, seed=2, max_epochs=3, augment=identity, **common
    )
    assert plain.history == augmented.history


def test_learning_rate_halves_when_val_stalls(tmp_path):
    from sonar_pipeline.cnn import PLATEAU_FACTOR

    common = toy_inputs(write_toy(tmp_path / "p"))
    run = train_run(
        lr=1e-3, weight_decay=0.0, seed=0, max_epochs=25, patience=25, **common
    )
    rates = sorted({round(h["lr"], 12) for h in run.history}, reverse=True)
    assert rates[0] == pytest.approx(1e-3)
    assert len(rates) >= 2 and rates[1] == pytest.approx(1e-3 * PLATEAU_FACTOR)


def test_same_seed_gives_the_same_run_on_cuda(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA here: run in the GPU image (compose.gpu.yaml)")
    common = toy_inputs(write_toy(tmp_path / "p"))
    device = torch.device("cuda")
    common["device"] = device
    common["tensors"] = {k: v.to(device) for k, v in common["tensors"].items()}
    a = train_run(lr=1e-3, weight_decay=1e-2, seed=3, max_epochs=3, **common)
    b = train_run(lr=1e-3, weight_decay=1e-2, seed=3, max_epochs=3, **common)
    assert a.history == b.history
    for key in a.state:
        assert torch.equal(a.state[key], b.state[key])
