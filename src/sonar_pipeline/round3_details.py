"""Details of round 3 for the report (F5-T08): docs/results/round3_details.json.

round3.json keeps the scores per partition; this file adds what the agreed figures need and
otherwise lives only in outputs/round3/ (not versioned) or in the saved weights:

- speed and cost of every model (latency, segments and tokens per second, GPU memory, epochs,
  weight file size);
- mean learning curves of the final trainings;
- generalisation gap: macro F1 of the seed ensemble on train, val and test of every partition
  (train and val by inference with the saved weights; test from round3.json, not rescored);
- variance due to the partition and due to the seed;
- recording-level test results per vessel and agreement between models;
- calibration (reliability, ECE, Brier score, log loss) and one-vs-rest ROC of the pooled test
  segments.

Nothing is trained; test probabilities are read from the ensemble files written by the round.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from sonar_pipeline import baseline as linear
from sonar_pipeline.cnn import predict_proba, set_determinism
from sonar_pipeline.cv import (
    MODELS,
    SELECTION_SEED,
    Round3,
    build_network,
    calibration,
    family,
)
from sonar_pipeline.dataset import CLASSES, baseline_features
from sonar_pipeline.interpret_cv import Rows
from sonar_pipeline.metrics import f1_scores, write_metrics

ROC_POINTS = 101


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def final_records(engine: Round3, model: str, config: str, seeds) -> list[dict]:
    """The per-training JSON of every final training (seed 0 is the selection run)."""
    records = []
    for partition in engine.partitions:
        for seed in seeds:
            stage = "selection" if seed == SELECTION_SEED else "final"
            _, path = engine.paths(model, stage, config, partition, seed)
            records.append(_load(path))
    return records


def speed(records: list[dict], weights_path: Path) -> dict:
    def mean(section, key):
        values = [r.get(section, {}).get(key) for r in records]
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None

    gpu = [r["training"].get("gpu_peak_allocated_bytes") for r in records]
    gpu = [g for g in gpu if g]
    # Data loading (F5-T10; absent in trainings recorded before it).
    loading = [r["training"].get("data_loading") for r in records]
    loading = [x for x in loading if x]
    shares = [
        x["minibatch_seconds_total"]
        / sum(h["train_seconds"] for h in r["training"]["history"])
        for x, r in zip(loading, records, strict=False)
        if x.get("minibatch_seconds_total") is not None
    ]
    return {
        "trainings": len(records),
        "training_seconds_mean": mean("training", "seconds"),
        "epochs_run_mean": mean("training", "epochs_run"),
        "best_epoch_mean": mean("training", "best_epoch"),
        "training_segments_per_second": mean("training", "segments_per_second"),
        "training_tokens_per_second": mean("training", "tokens_per_second"),
        "training_flops_mean": mean("training", "training_flops_estimate"),
        "inference_latency_seconds": mean("inference", "single_segment_seconds_median"),
        "inference_segments_per_second": mean("inference", "segments_per_second"),
        "gpu_peak_allocated_bytes_max": max(gpu) if gpu else None,
        "weights_bytes": weights_path.stat().st_size,
        "features_from_disk_seconds": loading[0]["features_from_disk_seconds_per_run"]
        if loading
        else None,
        "partition_seconds_mean": float(
            np.mean([x["partition_seconds"] for x in loading])
        )
        if loading
        else None,
        "minibatch_share_of_training": float(np.mean(shares)) if shares else None,
    }


def curves(records: list[dict]) -> dict | None:
    """Mean and std per epoch over the runs that reached it."""
    histories = [r["training"].get("history") for r in records]
    histories = [h for h in histories if h]
    if not histories:
        return None
    longest = max(len(h) for h in histories)
    out = {"epoch": list(range(1, longest + 1)), "runs": []}
    keys = ["train_loss", "val_loss", "val_f1_macro", "lr"]
    if all("grad_norm_mean" in h[0] for h in histories):  # F5-T10 onwards
        keys += ["grad_norm_mean", "grad_norm_max"]
    for key in keys:
        means, stds = [], []
        for epoch in range(longest):
            values = [h[epoch][key] for h in histories if len(h) > epoch]
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values)))
        out[key] = {"mean": means, "std": stds}
    out["runs"] = [sum(len(h) > e for h in histories) for e in range(longest)]
    out["best_epochs"] = [int(r["training"]["best_epoch"]) for r in records]
    return out


def ensemble_scores(
    engine, model, config_name, config, partition, seeds, roles
) -> dict:
    """Macro F1 of the seed ensemble on the given roles (no test here)."""
    pool = engine.pool
    shapes = pool.lofar.shape[1:] + (pool.demon.shape[1],)
    out = {}
    for role in roles:
        rows = engine.rows[partition.name][role]
        probs = []
        for seed in seeds:
            weights, _ = engine.paths(model, "final", config_name, partition, seed)
            if family(model) == "baseline":
                view = Rows(
                    pool.lofar[rows],
                    pool.demon[rows],
                    pool.labels[rows],
                    pool.vessel_group[rows],
                    pool.lofar_hz,
                )
                probs.append(
                    linear.predict_proba(
                        linear.load_baseline(weights), baseline_features(view)
                    )
                )
                continue
            net = build_network(model, config, shapes)
            net.load_state_dict(
                torch.load(weights, map_location="cpu", weights_only=True)
            )
            net.to(engine.device).eval()
            idx = torch.as_tensor(rows, device=engine.device)
            dp = engine.device_pool()
            probs.append(predict_proba(net, dp["lofar"][idx], dp["demon"][idx]))
        out[role] = f1_scores(pool.labels[rows], np.mean(probs, axis=0).argmax(axis=1))[
            0
        ]
    return out


def variance(per_partition: list[dict]) -> dict:
    """Between-partition variance of the seed mean and mean within-partition seed variance."""
    scores = np.array([p["test_f1_macro_per_seed"] for p in per_partition])  # (P, S)
    seeds = scores.shape[1]
    return {
        "seeds": seeds,
        "partition_std": float(scores.mean(axis=1).std(ddof=1)),
        "seed_std": float(np.sqrt(scores.var(axis=1, ddof=1).mean()))
        if seeds > 1
        else None,
    }


def roc_pooled(probs: np.ndarray, labels: np.ndarray) -> dict:
    grid = np.linspace(0, 1, ROC_POINTS)
    out = {}
    for k, name in enumerate(CLASSES):
        positive = labels == k
        fpr, tpr, _ = roc_curve(positive, probs[:, k])
        out[name] = {
            "pfa": grid.tolist(),
            "pd": np.interp(grid, fpr, tpr).tolist(),
            "auc": float(roc_auc_score(positive, probs[:, k])),
        }
    return out


def collect_details(engine: Round3, summary: dict, output: Path, log=print) -> dict:
    set_determinism(0)
    group_of = dict(
        zip(
            engine.pool.wav_path.tolist(),
            engine.pool.vessel_group.tolist(),
            strict=True,
        )
    )
    models, outcomes = {}, {}
    for model in MODELS:
        if model not in summary["models"]:
            continue
        entry = summary["models"][model]
        config_name, config = entry["chosen_config"], entry["chosen_hyperparameters"]
        seeds = [0] if family(model) == "baseline" else list(summary["seeds"])
        records = final_records(engine, model, config_name, seeds)
        first = engine.partitions[0]
        weights, _ = engine.paths(model, "final", config_name, first, SELECTION_SEED)

        generalisation, probs, labels = [], [], []
        vessel: dict[str, list[int]] = {}
        outcome = {}
        for partition, row in zip(
            engine.partitions, entry["per_partition"], strict=True
        ):
            scores = ensemble_scores(
                engine, model, config_name, config, partition, seeds, ("train", "val")
            )
            generalisation.append(
                {
                    "partition": partition.name,
                    "train_f1_macro": scores["train"],
                    "val_f1_macro": scores["val"],
                    "test_f1_macro": row["test_f1_macro_segment"],
                }
            )
            ensemble = _load(
                engine.outputs_dir / model / "final" / partition.name / "ensemble.json"
            )
            probs.append(np.array(ensemble["test_probabilities"]))
            labels.append(engine.pool.labels[engine.rows[partition.name]["test"]])
            for rec in ensemble["test"]["recordings"]:
                correct = int(rec["predicted"] == rec["label"])
                vessel.setdefault(group_of[rec["wav_path"]], []).append(correct)
                outcome[(partition.name, rec["wav_path"])] = rec["predicted"]
            log(f"[{model}] {partition.name}: details collected")
        probs, labels = np.concatenate(probs), np.concatenate(labels)
        p_true = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)
        onehot = np.eye(len(CLASSES))[labels]
        models[model] = {
            "speed": speed(records, weights),
            "curves": curves(records),
            "generalisation": generalisation,
            "variance": variance(entry["per_partition"]),
            "vessels": {
                g: {
                    "tests": len(v),
                    "correct": int(sum(v)),
                    "accuracy": float(np.mean(v)),
                }
                for g, v in sorted(vessel.items())
            },
            "calibration": {
                **calibration(probs, labels),
                "brier": float(((probs - onehot) ** 2).sum(axis=1).mean()),
                "log_loss": float(-np.log(p_true).mean()),
            },
            "roc": roc_pooled(probs, labels),
        }
        outcomes[model] = outcome

    truth = {}
    for model in outcomes:
        for partition in engine.partitions:
            ensemble = _load(
                engine.outputs_dir / model / "final" / partition.name / "ensemble.json"
            )
            for rec in ensemble["test"]["recordings"]:
                truth[(partition.name, rec["wav_path"])] = rec["label"]
        break
    names = list(outcomes)
    agreement = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            keys = sorted(outcomes[a])
            same = np.mean([outcomes[a][k] == outcomes[b][k] for k in keys])
            both_wrong = np.mean(
                [
                    outcomes[a][k] != truth[k] and outcomes[b][k] != truth[k]
                    for k in keys
                ]
            )
            either = np.mean(
                [outcomes[a][k] == truth[k] or outcomes[b][k] == truth[k] for k in keys]
            )
            agreement[f"{a}|{b}"] = {
                "same_prediction": float(same),
                "both_wrong": float(both_wrong),
                "at_least_one_right": float(either),
                "cases": len(keys),
            }
    report = {
        "round": 3,
        "note": "train and val F1 by inference with the saved final weights; test values and "
        "probabilities come from the round (not rescored)",
        "models": models,
        "agreement_recording_level": agreement,
        "device": str(engine.device),
        "hardware": engine.hardware,
        "code_sha256": engine.code,
        "round3_code_sha256": summary["code_sha256"],
        "cv_splits_sha256": engine.splits_sha256,
    }
    write_metrics(output, report)
    return report
