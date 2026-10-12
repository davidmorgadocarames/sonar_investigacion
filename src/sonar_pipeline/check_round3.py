"""Quick check of an installation against the round 3 results (F5-T11), on CPU in a few seconds.

The 7 representative models versioned in models/round3/representative/ (partition rep0_fold0,
seed 0, chosen configuration) score the test segments of that partition, and their macro F1 is
compared with the one recorded for that seed in docs/results/round3.json. Nothing is trained or
chosen here: it only rescores what the round already scored, to show that the features, the code
and the weights give the same numbers. The weights were trained on a GPU, so on a CPU the last
digit may change; the tolerance accepts it.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sonar_pipeline import baseline as linear
from sonar_pipeline.cnn import predict_proba
from sonar_pipeline.cv import (
    MODELS,
    build_network,
    family,
    load_pool,
    partition_rows,
    read_partitions,
)
from sonar_pipeline.dataset import baseline_features
from sonar_pipeline.interpret_cv import Rows
from sonar_pipeline.metrics import f1_scores

# The weights were trained on a GPU; on a CPU a few test segments (of about 380) may change
# class, and each one moves the macro F1 by about 0.003: 0.01 accepts two or three of them.
TOLERANCE = 0.01


def check_round3(
    processed_dir: Path,
    splits_path: Path,
    representative_dir: Path,
    summary: dict,
    tolerance: float = TOLERANCE,
) -> dict:
    """{model: {f1, reference, difference, weights_ok}} and the list of problems found."""
    torch.set_num_threads(4)
    pool = load_pool(processed_dir)
    partitions = {p.name: p for p in read_partitions(splits_path)}
    shapes = pool.lofar.shape[1:] + (pool.demon.shape[1],)
    results, problems = {}, []
    for model in MODELS:
        if model not in summary["models"]:
            continue
        info_path = representative_dir / f"{model}.json"
        if not info_path.is_file():
            problems.append(f"{model}: {info_path} not found")
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        suffix = ".npz" if family(model) == "baseline" else ".pt"
        weights = representative_dir / f"{model}{suffix}"
        weights_ok = (
            hashlib.sha256(weights.read_bytes()).hexdigest() == info["weights_sha256"]
        )
        if (
            not weights_ok
        ):  # not the weights the reference was measured with: not scored
            problems.append(f"{model}: weights differ from the versioned ones")
            continue

        partition = partitions[info["partition"]]
        rows = partition_rows(pool, partition)["test"]
        labels = pool.labels[rows]
        if family(model) == "baseline":
            view = Rows(
                pool.lofar[rows],
                pool.demon[rows],
                labels,
                pool.vessel_group[rows],
                pool.lofar_hz,
            )
            probs = linear.predict_proba(
                linear.load_baseline(weights), baseline_features(view)
            )
        else:
            net = build_network(model, info["hyperparameters"], shapes)
            net.load_state_dict(
                torch.load(weights, map_location="cpu", weights_only=True)
            )
            net.eval()
            probs = predict_proba(
                net,
                torch.as_tensor(pool.lofar[rows]),
                torch.as_tensor(pool.demon[rows]),
            )
        f1 = f1_scores(labels, np.asarray(probs).argmax(axis=1))[0]

        entry = summary["models"][model]
        row = next(
            p
            for p in entry["per_partition"]
            if (p["repetition"], p["fold"]) == (partition.repetition, partition.fold)
        )
        seeds = [0] if family(model) == "baseline" else list(summary["seeds"])
        reference = row["test_f1_macro_per_seed"][seeds.index(info["seed"])]
        difference = f1 - reference
        if abs(difference) > tolerance:
            problems.append(f"{model}: test F1 {f1:.4f} instead of {reference:.4f}")
        results[model] = {
            "partition": partition.name,
            "seed": info["seed"],
            "test_f1_macro": f1,
            "reference": reference,
            "difference": difference,
            "weights_ok": weights_ok,
        }
    return {"models": results, "problems": problems, "tolerance": tolerance}
