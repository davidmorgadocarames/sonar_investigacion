"""Round 3 engine tests (F5-T03) on toy data, on CPU. No real training here."""

import csv
import itertools
import json

import numpy as np
import pytest
import torch

from sonar_pipeline.__main__ import main
from sonar_pipeline.cv import (
    BLOCKS,
    CONFIGS,
    MODELS,
    WARMUP_EPOCHS,
    Round3,
    calibration,
    fit_network,
    holm,
    load_pool,
    nadeau_bengio,
    partition_rows,
    read_partitions,
    split_parameters,
    warmup_lr,
)
from sonar_pipeline.dataset import CLASSES
from sonar_pipeline.transformer import SonarTransformer

FIELDS = ["segment_id", "wav_path", "vessel_class", "vessel_group", "split"]
CV_FIELDS = ["repetition", "fold", "wav_path", "vessel_class", "vessel_group", "split"]
FRAMES, LOFAR_BINS, DEMON_BINS = 4, 101, 40
VESSELS, SEGMENTS = 6, 4

# Two partitions: vessel `test` in test, `val` in val, the other 4 in train (every class).
TOY_PARTITIONS = ((0, 0, 0, 1), (0, 1, 2, 3))
# Small configurations so the toy runs in seconds.
TOY_CONFIGS = {
    "baseline": {"A": {"c": 0.01}, "B": {"c": 1.0}},
    "cnn": {"A": CONFIGS["cnn"]["A"], "B": CONFIGS["cnn"]["B"]},
    "transformer": {"A": CONFIGS["transformer"]["A"]},
    "hybrid": {"A": CONFIGS["hybrid"]["A"]},
}


def write_round3_toy(folder, seed=0, scramble_vessel=None):
    """4 classes x 6 vessels x 4 segments in features_{train,val,test}.npz, plus cv_splits.csv.

    Class k has a tonal line at LOFAR bin 10 + 20 k and a DEMON peak at bin 4 + 8 k. With
    scramble_vessel=v, the segments of vessel v of every class are replaced by noise.
    """
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    index, lofar, demon = [], [], []
    for k, cls in enumerate(CLASSES):
        for v in range(VESSELS):
            split = ("train", "train", "train", "train", "val", "test")[v]
            for _ in range(SEGMENTS):
                index.append((len(index), f"{cls}/{v}.wav", cls, f"{cls}/V{v}", split))
                frame = rng.normal(size=(FRAMES, LOFAR_BINS))
                spectrum = rng.random(DEMON_BINS) * 0.3
                if v != scramble_vessel:
                    frame[:, 10 + 20 * k] += 6.0
                    spectrum[4 + 8 * k] = 1.0
                lofar.append(frame)
                demon.append(spectrum)
    with (folder / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows(index)
    lofar, demon = np.array(lofar, np.float32), np.array(demon, np.float32)
    for split in ("train", "val", "test"):
        rows = [r[0] for r in index if r[4] == split]
        np.savez(
            folder / f"features_{split}.npz",
            segment_id=np.array(rows),
            lofar=lofar[rows],
            demon=demon[rows],
            lofar_hz=np.arange(float(LOFAR_BINS)) * 2,
            demon_hz=np.arange(float(DEMON_BINS)) / 3,
        )
    (folder / "features_meta.json").write_text(
        json.dumps({"split_sha256": "toy", "code_sha256": "toy"}), encoding="utf-8"
    )
    splits = folder / "cv_splits.csv"
    with splits.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(CV_FIELDS)
        for rep, fold, test, val in TOY_PARTITIONS:
            for cls in CLASSES:
                for v in range(VESSELS):
                    role = "test" if v == test else "val" if v == val else "train"
                    writer.writerow(
                        [rep, fold, f"{cls}/{v}.wav", cls, f"{cls}/V{v}", role]
                    )
    return folder


def engine(tmp_path, folder=None, **kwargs):
    folder = folder or write_round3_toy(tmp_path / "p")
    options = {
        "max_epochs": 3,
        "patience": 3,
        "n_bootstrap": 10,
        "configs": TOY_CONFIGS,
    }
    options = {**options, "device_name": "cpu", "log": lambda line: None, **kwargs}
    return Round3(
        folder,
        folder / "cv_splits.csv",
        tmp_path / "models" / "round3",
        tmp_path / "outputs",
        **options,
    )


# --------------------------------------------------------------------------- data


def test_pool_joins_the_three_splits_in_segment_order(tmp_path):
    pool = load_pool(write_round3_toy(tmp_path / "p"))
    assert len(pool) == len(CLASSES) * VESSELS * SEGMENTS
    assert np.array_equal(pool.segment_id, np.arange(len(pool)))
    assert pool.lofar.shape == (len(pool), FRAMES, LOFAR_BINS)


def test_partition_rows_follow_the_csv(tmp_path):
    folder = write_round3_toy(tmp_path / "p")
    pool = load_pool(folder)
    partitions = read_partitions(folder / "cv_splits.csv")
    assert [p.name for p in partitions] == ["rep0_fold0", "rep0_fold1"]
    rows = partition_rows(pool, partitions[0])
    assert set(pool.vessel_group[rows["test"]]) == {f"{c}/V0" for c in CLASSES}
    assert set(pool.vessel_group[rows["val"]]) == {f"{c}/V1" for c in CLASSES}
    assert sum(len(r) for r in rows.values()) == len(pool)


def test_a_vessel_in_two_roles_is_rejected(tmp_path):
    folder = write_round3_toy(tmp_path / "p")
    pool = load_pool(folder)
    partition = read_partitions(folder / "cv_splits.csv")[0]
    pool.vessel_group[pool.wav_path == "Cargo/0.wav"] = "Cargo/V2"  # test and train
    with pytest.raises(ValueError, match="Cargo/V2"):
        partition_rows(pool, partition)


def test_a_role_without_a_class_is_rejected(tmp_path):
    folder = write_round3_toy(tmp_path / "p")
    pool = load_pool(folder)
    partition = read_partitions(folder / "cv_splits.csv")[0]
    partition.roles["Tug/0.wav"] = "train"  # Tug loses its only test vessel
    with pytest.raises(ValueError, match="test has no"):
        partition_rows(pool, partition)


# --------------------------------------------------------------------------- statistics


def test_nadeau_bengio_matches_the_formula():
    a = np.array([0.50, 0.55, 0.60, 0.52, 0.58])
    b = np.array([0.45, 0.50, 0.50, 0.50, 0.50])
    d = a - b
    ratio = 0.25
    expected_t = d.mean() / np.sqrt((1 / 5 + ratio) * d.var(ddof=1))
    result = nadeau_bengio(a, b, ratio)
    assert result["t"] == pytest.approx(expected_t)
    assert result["df"] == 4
    assert result["wins"] == 5 and result["losses"] == 0
    from scipy import stats

    assert result["p"] == pytest.approx(2 * stats.t.sf(expected_t, 4))
    # The correction makes the test more conservative than the naive paired t-test.
    assert result["p"] > stats.ttest_rel(a, b).pvalue


def test_nadeau_bengio_with_identical_scores():
    result = nadeau_bengio([0.4, 0.5], [0.4, 0.5], 0.3)
    assert result["p"] == 1.0 and result["mean_difference"] == 0.0


def test_holm_adjusts_in_input_order():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_calibration_of_a_perfect_and_an_overconfident_model():
    labels = np.array([0, 1, 2, 3])
    perfect = np.eye(4)
    assert calibration(perfect, labels)["ece"] == pytest.approx(0.0)
    wrong = np.eye(4)[[1, 2, 3, 0]]  # always sure and always wrong
    assert calibration(wrong, labels)["ece"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- training


def test_warmup_is_linear_from_ten_percent():
    assert warmup_lr(1.0, 1) == pytest.approx(0.1)
    assert warmup_lr(1.0, 3) == pytest.approx(0.1 + 0.9 * 2 / WARMUP_EPOCHS)
    assert warmup_lr(1.0, WARMUP_EPOCHS + 1) == 1.0


def test_no_weight_decay_on_norms_biases_and_class_token():
    model = SonarTransformer(
        lofar_frames=FRAMES, lofar_bins=LOFAR_BINS, demon_bins=DEMON_BINS
    )
    decay, no_decay = split_parameters(model)
    no_decay_ids = {id(p) for p in no_decay["params"]}
    assert id(model.cls_token) in no_decay_ids
    assert id(model.norm.weight) in no_decay_ids
    assert id(model.lofar_embed.bias) in no_decay_ids
    assert id(model.lofar_embed.weight) in {id(p) for p in decay["params"]}
    assert no_decay["weight_decay"] == 0.0


def toy_data(tmp_path):
    e = engine(tmp_path)
    return e, e.net_data(e.partitions[0])


def test_the_transformer_warms_up_then_keeps_the_lr(tmp_path):
    e, data = toy_data(tmp_path)
    config = CONFIGS["transformer"]["A"]
    _, run = fit_network(
        "transformer", config, data, 0, e.device, max_epochs=6, patience=6
    )
    lrs = [h["lr"] for h in run.history]
    assert lrs[:5] == pytest.approx([warmup_lr(config["lr"], k) for k in range(1, 6)])
    # After warmup the lr is the configured one (F5-T10: it used to stay at 82 %).
    assert lrs[5] == config["lr"]


def test_after_warmup_the_lr_only_changes_by_the_plateau_rule(tmp_path):
    e, data = toy_data(tmp_path)
    config = {**CONFIGS["transformer"]["A"], "lr": 1e-3}
    _, run = fit_network(
        "transformer", config, data, 0, e.device, max_epochs=20, patience=20
    )
    after = [h["lr"] for h in run.history[WARMUP_EPOCHS:]]
    assert after[0] == config["lr"]
    for previous, current in itertools.pairwise(after):
        assert current in (previous, pytest.approx(previous * 0.5))


def test_every_epoch_records_gradient_norm_and_data_time(tmp_path):
    e, data = toy_data(tmp_path)
    _, run = fit_network(
        "cnn_aug",
        CONFIGS["cnn"]["A"],
        data,
        0,
        e.device,
        e.augment_for("cnn_aug", e.partitions[0]),
        max_epochs=2,
    )
    for h in run.history:
        assert h["grad_norm_max"] >= h["grad_norm_mean"] > 0
        assert 0 <= h["data_seconds"] <= h["train_seconds"]


def test_the_cnn_has_no_warmup(tmp_path):
    e, data = toy_data(tmp_path)
    config = CONFIGS["cnn"]["A"]
    _, run = fit_network("cnn", config, data, 0, e.device, max_epochs=2, patience=2)
    assert run.history[0]["lr"] == config["lr"]


def test_every_configuration_builds_and_runs(tmp_path):
    e, data = toy_data(tmp_path)
    for model in MODELS:
        family = model.removesuffix("_aug")
        if family == "baseline":
            continue
        for config in CONFIGS[family].values():
            _net, run = fit_network(model, config, data, 0, e.device, max_epochs=1)
            assert run.epochs_run == 1 and run.state is not None


# --------------------------------------------------------------------------- the round


def test_round_trains_selects_ensembles_and_compares(tmp_path):
    e = engine(tmp_path, seeds=(0, 1))
    summary = e.run(("baseline", "cnn", "transformer"), tmp_path / "round3.json")
    assert set(summary["models"]) == {"baseline", "cnn", "transformer"}
    assert summary["failures"] == {}
    cnn = summary["models"]["cnn"]
    assert cnn["chosen_config"] in ("A", "B")
    assert len(cnn["per_partition"]) == 2
    assert len(cnn["per_partition"][0]["test_f1_macro_per_seed"]) == 2
    assert (
        cnn["selection"]["A"]["per_partition"]
        and cnn["selection"]["B"]["per_partition"]
    )
    # Only the comparisons whose two models ran.
    pairs = {(c["model"], c["against"]) for c in summary["comparisons"]}
    assert pairs == {
        ("cnn", "baseline"),
        ("transformer", "baseline"),
        ("transformer", "cnn"),
    }
    assert all(c["p_holm"] >= c["p"] for c in summary["comparisons"])
    saved = json.loads((tmp_path / "round3.json").read_text(encoding="utf-8"))
    assert saved["models"]["transformer"]["cost"]["tokens_per_segment"] == 2 + 5
    # Weights: every training in its folder, seed 0 copied to final, one representative.
    root = tmp_path / "models" / "round3"
    assert (root / "cnn" / "selection" / "B" / "rep0_fold1.pt").is_file()
    assert (root / "cnn" / "final" / "rep0_fold0" / "seed0.pt").is_file()
    assert (root / "cnn" / "final" / "rep0_fold0" / "seed1.pt").is_file()
    assert (root / "representative" / "cnn.pt").is_file()
    assert (root / "representative" / "baseline.npz").is_file()


def test_training_json_records_data_loading_and_threads(tmp_path):
    e = engine(tmp_path, seeds=(0,))
    summary = e.run(("baseline", "cnn"))
    out = tmp_path / "outputs"
    net = json.loads(
        (out / "cnn" / "selection" / "A" / "rep0_fold0.json").read_text("utf-8")
    )
    loading = net["training"]["data_loading"]
    assert loading["features_from_disk_seconds_per_run"] > 0
    assert loading["partition_seconds"] >= 0
    assert loading["minibatch_seconds_total"] >= 0
    base = json.loads(
        (out / "baseline" / "selection" / "A" / "rep0_fold0.json").read_text("utf-8")
    )
    assert base["training"]["data_loading"]["partition_seconds"] > 0
    assert summary["hardware"]["torch_threads"] == 4
    starts = [m["started"] for m in summary["models"].values()]
    assert summary["started"] == min(starts)


def test_per_training_json_has_the_agreed_data(tmp_path):
    e = engine(tmp_path)
    e.run_model("transformer")
    record = json.loads(
        (
            tmp_path / "outputs" / "transformer" / "final" / "rep0_fold1" / "seed1.json"
        ).read_text(encoding="utf-8")
    )
    assert record["complete"] and record["stage"] == "final"
    assert record["mlops"]["code_sha256"] and record["mlops"]["weights_sha256"]
    assert record["mlops"]["hardware"]["device"] == "cpu"
    assert record["cost"]["mflops_per_segment"] > 0
    training = record["training"]
    assert training["tokens_per_second"] > 0 and training["segments_per_second"] > 0
    assert training["training_flops_estimate"] > 0
    assert {"train_loss", "val_loss", "val_f1_macro", "lr", "seconds"} <= set(
        training["history"][0]
    )
    assert record["inference"]["single_segment_seconds_median"] > 0
    assert record["result"]["val"]["calibration"]["bins"]
    assert len(record["result"]["val"]["recordings"]) == len(CLASSES)
    data = record["partition"]["data"]
    assert data["test"]["per_class"]["Tug"]["vessels"] == ["Tug/V2"]
    assert "test" not in record["result"]
    ensemble = json.loads(
        (
            tmp_path
            / "outputs"
            / "transformer"
            / "final"
            / "rep0_fold1"
            / "ensemble.json"
        ).read_text(encoding="utf-8")
    )
    assert len(ensemble["test_probabilities"]) == len(CLASSES) * SEGMENTS


def test_finished_trainings_are_not_repeated(tmp_path):
    e = engine(tmp_path)
    e.run_model("cnn")
    path = tmp_path / "outputs" / "cnn" / "selection" / "A" / "rep0_fold0.json"
    first = path.read_text(encoding="utf-8")
    calls = []
    original = e._train_network
    e._train_network = lambda *a, **k: calls.append(a) or original(*a, **k)
    e.run_model("cnn")
    assert calls == [] and path.read_text(encoding="utf-8") == first
    # A different configuration is trained again.
    e.configs = {**TOY_CONFIGS, "cnn": {"A": {**CONFIGS["cnn"]["A"], "dropout": 0.4}}}
    e.run_model("cnn")
    assert len(calls) == 2 + 2 * 2  # seed 0 in selection, seeds 1 and 2 in final


def test_selection_never_touches_test(tmp_path, monkeypatch):
    e = engine(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("test used during selection")

    monkeypatch.setattr(e, "test_tensors", forbidden)
    monkeypatch.setattr(e, "predict_test", forbidden)
    for model in ("baseline", "cnn", "transformer", "hybrid"):
        for config_name in TOY_CONFIGS[model]:
            for partition in e.partitions:
                e.train_one(model, "selection", config_name, partition, 0)


def test_changing_test_does_not_change_the_selection(tmp_path):
    """Vessel 0 is test in rep0_fold0: replacing it by noise must leave that partition's
    selection exactly the same, and change its test score."""
    a = engine(tmp_path / "a", write_round3_toy(tmp_path / "a" / "p"))
    b = engine(
        tmp_path / "b", write_round3_toy(tmp_path / "b" / "p", scramble_vessel=0)
    )
    part_a, part_b = a.partitions[0], b.partitions[0]
    for model in ("baseline", "cnn"):
        for config_name in TOY_CONFIGS[model]:
            ra = a.train_one(model, "selection", config_name, part_a, 0)
            rb = b.train_one(model, "selection", config_name, part_b, 0)
            assert (
                ra["result"]["best_val_f1_macro"] == rb["result"]["best_val_f1_macro"]
            )
            assert ra["mlops"]["weights_sha256"] == rb["mlops"]["weights_sha256"]


def test_a_failing_model_does_not_stop_the_others(tmp_path):
    broken = {
        **TOY_CONFIGS,
        "transformer": {"A": {**CONFIGS["transformer"]["A"], "heads": 3}},
    }
    e = engine(tmp_path, configs=broken, seeds=(0,))
    summary = e.run(("baseline", "transformer", "hybrid"))
    assert set(summary["models"]) == {"baseline", "hybrid"}
    assert "transformer" in summary["failures"]
    assert (tmp_path / "outputs" / "transformer" / "error.txt").is_file()


def test_blocks_cover_every_model_once():
    assert sorted(m for block in BLOCKS for m in block) == sorted(MODELS)


def test_measure_reports_time_per_configuration(tmp_path):
    e = engine(tmp_path)
    result = e.measure(epochs=1)
    assert set(result) == {
        f"{m}/{c}"
        for m in MODELS
        if m != "baseline"
        for c in TOY_CONFIGS[m.removesuffix("_aug")]
    }
    assert all(v["seconds_per_epoch"] > 0 for v in result.values())


# --------------------------------------------------------------------------- command line


def test_cli_runs_round_3(tmp_path, capsys):
    folder = write_round3_toy(tmp_path / "p")
    argv = ["train", "--round", "3", "--only", "baseline", "cnn", "--max-epochs", "1"]
    argv += ["--processed-dir", str(folder), "--splits", str(folder / "cv_splits.csv")]
    argv += ["--models-dir", str(tmp_path / "m"), "--outputs-dir", str(tmp_path / "o")]
    argv += [
        "--summary",
        str(tmp_path / "round3.json"),
        "--device",
        "cpu",
        "--bootstrap",
        "5",
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "cnn vs baseline" in out
    summary = json.loads((tmp_path / "round3.json").read_text(encoding="utf-8"))
    assert summary["models"]["cnn"]["selection"].keys() == CONFIGS["cnn"].keys()
    assert (tmp_path / "o" / "run.log").is_file()
    assert (tmp_path / "m" / "round3" / "representative" / "cnn.pt").is_file()


def test_cli_needs_model_or_round(capsys):
    assert main(["train"]) == 1
    assert main(["train", "--model", "baseline", "--round", "3"]) == 1
    assert main(["train", "--model", "baseline", "--measure"]) == 1


def test_seed_zero_reuse_equals_a_new_training(tmp_path):
    """Final seed 0 reuses the selection run: retraining gives the same weights."""
    e, data = toy_data(tmp_path)
    config = CONFIGS["hybrid"]["A"]
    _, a = fit_network(
        "hybrid_aug",
        config,
        data,
        0,
        e.device,
        e.augment_for("hybrid_aug", e.partitions[0]),
        max_epochs=2,
    )
    _, b = fit_network(
        "hybrid_aug",
        config,
        data,
        0,
        e.device,
        e.augment_for("hybrid_aug", e.partitions[0]),
        max_epochs=2,
    )
    for key in a.state:
        assert torch.equal(a.state[key], b.state[key])


def test_the_summary_collects_models_of_earlier_blocks(tmp_path):
    e = engine(tmp_path, seeds=(0,))
    e.run(("baseline",))
    summary = e.run(("cnn",))
    assert list(summary["models"]) == ["baseline", "cnn"]
    assert [(c["model"], c["against"]) for c in summary["comparisons"]] == [
        ("cnn", "baseline")
    ]
    # A summary from other code is not mixed in.
    e.code = "other"
    assert list(e.run(("transformer",))["models"]) == ["transformer"]


def test_the_plateau_rule_waits_for_the_end_of_warmup(tmp_path, monkeypatch):
    """ReduceLROnPlateau only sees the epochs after warmup (F5-T10, reviewer)."""
    calls = []
    original = torch.optim.lr_scheduler.ReduceLROnPlateau.step

    def step(self, metrics, *args, **kwargs):
        calls.append(metrics)
        return original(self, metrics, *args, **kwargs)

    monkeypatch.setattr(torch.optim.lr_scheduler.ReduceLROnPlateau, "step", step)
    e, data = toy_data(tmp_path)
    _, run = fit_network(
        "transformer",
        CONFIGS["transformer"]["A"],
        data,
        0,
        e.device,
        max_epochs=7,
        patience=7,
    )
    assert len(calls) == run.epochs_run - WARMUP_EPOCHS == 2
    assert calls == [h["val_f1_macro"] for h in run.history[WARMUP_EPOCHS:]]
    calls.clear()
    fit_network("cnn", CONFIGS["cnn"]["A"], data, 0, e.device, max_epochs=3, patience=3)
    assert len(calls) == 3  # no warmup: from the first epoch
