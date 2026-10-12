"""Interpretability of round 3 (F5-T07, D-024), on val with the saved final models; no test.

For every model and every partition, the chosen configuration is loaded with its final seeds
(0, 1 and 2; the baseline has one fit) and measured on that partition's val segments:

- Baseline: permutation importance of each band (as in F3-T10): drop in val macro F1 when the
  band is shuffled across segments.
- Networks: occlusion of each band (set to the partition's train mean, stored in the weights):
  drop in the probability of the true class; and saliency |d logit(true class) / d input|,
  averaged per band and class.
- Transformer and hybrid: attention of the class token to the DEMON token and to each frequency
  token, averaged over heads, segments and seeds, per layer. Hybrid positions are mapped to the
  40 Hz bands by their centre frequency.

The seeds are averaged inside each partition; the result is the mean and standard deviation
over the partitions. The era check of F3-T10 (DT-005) is repeated with the vessels of the
whole pool: do the LOFAR bands a model relies on coincide with the ones that differ between the
2016 and 2017 recordings?
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sonar_pipeline import baseline as linear
from sonar_pipeline.cnn import predict_proba, set_determinism
from sonar_pipeline.cv import MODELS, Round3, build_network, family
from sonar_pipeline.dataset import CLASSES, baseline_features
from sonar_pipeline.interpret import (
    DEMON_BAND_HZ,
    LOFAR_BAND_HZ,
    REPEATS,
    SEED,
    TOP,
    bands,
    era_check,
)
from sonar_pipeline.metrics import f1_scores, write_metrics

SALIENCY_BATCH = 64


@dataclass
class Rows:
    """The fields era_check and baseline_features read."""

    lofar: np.ndarray
    demon: np.ndarray
    labels: np.ndarray
    vessel_group: np.ndarray
    lofar_hz: np.ndarray


# --------------------------------------------------------------------------- measures


def permutation_importance(weights, val: Rows, lofar_bands, demon_bands, seed=SEED):
    """Drop in val macro F1 per band of the baseline (LOFAR bands, then DEMON bands)."""
    x = baseline_features(val)
    n_lofar = val.lofar.shape[2]
    base, _ = f1_scores(val.labels, linear.predict_proba(weights, x).argmax(axis=1))
    rng = np.random.default_rng(seed)
    columns = [b["bins"] for b in lofar_bands] + [
        n_lofar + b["bins"] for b in demon_bands
    ]
    drops = []
    for cols in columns:
        values = []
        for _ in range(REPEATS):
            shuffled = x.copy()
            shuffled[:, cols] = x[rng.permutation(len(x))][:, cols]
            f1, _ = f1_scores(
                val.labels, linear.predict_proba(weights, shuffled).argmax(axis=1)
            )
            values.append(base - f1)
        drops.append(np.mean(values))
    n = len(lofar_bands)
    return np.array(drops[:n]), np.array(drops[n:])


def true_probability(model, lofar, demon, labels: np.ndarray) -> np.ndarray:
    probs = predict_proba(model, lofar, demon)
    return probs[np.arange(len(labels)), labels]


def occlusion(model, lofar, demon, labels, lofar_bands, demon_bands):
    """Drop in the true-class probability per band: overall (bands,) and per class."""
    base = true_probability(model, lofar, demon, labels)
    lofar_fill, demon_fill = float(model.lofar_mean), float(model.demon_mean)

    def drops(branch, band_list):
        overall, per_class = [], []
        for band in band_list:
            idx = torch.as_tensor(band["bins"], device=lofar.device)
            a, d = lofar.clone(), demon.clone()
            if branch == "lofar":
                a[:, :, idx] = lofar_fill
            else:
                d[:, idx] = demon_fill
            drop = base - true_probability(model, a, d, labels)
            overall.append(drop.mean())
            per_class.append([drop[labels == k].mean() for k in range(len(CLASSES))])
        return np.array(overall), np.array(per_class)

    lofar_drop, lofar_class = drops("lofar", lofar_bands)
    demon_drop, demon_class = drops("demon", demon_bands)
    return lofar_drop, demon_drop, lofar_class, demon_class


def saliency(model, lofar, demon, labels, lofar_bands, demon_bands):
    """Mean |d logit(true class) / d input| per class and band: LOFAR (classes, bands), DEMON."""
    lofar_sum = torch.zeros(len(CLASSES), lofar.shape[2], device=lofar.device)
    demon_sum = torch.zeros(len(CLASSES), demon.shape[1], device=lofar.device)
    y = torch.as_tensor(labels, device=lofar.device)
    model.eval()
    for start in range(0, len(labels), SALIENCY_BATCH):
        a = lofar[start : start + SALIENCY_BATCH].clone().requires_grad_(True)
        d = demon[start : start + SALIENCY_BATCH].clone().requires_grad_(True)
        rows = y[start : start + SALIENCY_BATCH]
        model(a, d).gather(1, rows[:, None]).sum().backward()
        one_hot = (
            torch.nn.functional.one_hot(rows, len(CLASSES)).float().T
        )  # (classes, B)
        lofar_sum += one_hot @ a.grad.abs().mean(dim=1)  # mean over frames
        demon_sum += one_hot @ d.grad.abs()
    counts = np.maximum(np.bincount(labels, minlength=len(CLASSES)), 1)[:, None]
    lofar_map = lofar_sum.cpu().numpy() / counts
    demon_map = demon_sum.cpu().numpy() / counts
    return (
        np.array([[row[b["bins"]].mean() for b in lofar_bands] for row in lofar_map]),
        np.array([[row[b["bins"]].mean() for b in demon_bands] for row in demon_map]),
    )


def class_token_attention(model, lofar, demon, batch: int = 256) -> np.ndarray:
    """Attention of the class token to every token, (layers, tokens), mean over heads and
    segments; token 0 is the class token, 1 the DEMON token, then the frequency tokens."""
    model.eval()
    model.keep_attention(True)
    total, n = None, 0
    with torch.no_grad():
        for start in range(0, len(lofar), batch):
            model(lofar[start : start + batch], demon[start : start + batch])
            rows = torch.stack(
                [m[:, :, 0, :].mean(dim=1) for m in model.attention_maps()]
            )
            part = rows.sum(dim=1)  # (layers, tokens)
            total = part if total is None else total + part
            n += rows.shape[1]
    model.keep_attention(False)
    return (total / n).cpu().numpy()


def token_centres_hz(
    model_name: str, n_tokens: int, lofar_hz: np.ndarray
) -> np.ndarray:
    """Centre frequency of every frequency token."""
    bin_hz = float(lofar_hz[1] - lofar_hz[0])
    if family(model_name) == "transformer":
        width = 20  # bins per token
    else:  # hybrid: two max-poolings of 4
        width = 16
    return (np.arange(n_tokens) * width + (width - 1) / 2) * bin_hz


def attention_by_band(
    attention: np.ndarray, centres: np.ndarray, lofar_bands
) -> np.ndarray:
    """Frequency-token attention summed into the 40 Hz bands (by centre frequency)."""
    out = np.zeros(len(lofar_bands))
    for value, centre in zip(attention, centres, strict=True):
        for i, band in enumerate(lofar_bands):
            if band["lo"] <= centre < band["hi"] or (
                i == len(lofar_bands) - 1 and centre >= band["lo"]
            ):
                out[i] += value
                break
    return out


# --------------------------------------------------------------------------- run


def stats(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": values.mean(axis=0),
        "std": values.std(axis=0, ddof=1)
        if len(values) > 1
        else np.zeros(values.shape[1:]),
    }


def top_bands(values: np.ndarray, band_list, n: int = TOP) -> list[dict]:
    order = np.argsort(-values)[:n]
    return [
        {
            "lo_hz": band_list[i]["lo"],
            "hi_hz": band_list[i]["hi"],
            "value": float(values[i]),
        }
        for i in order
    ]


def run_round3_interpretability(
    engine: Round3,
    round3_summary: dict,
    output_path: Path,
    models: tuple[str, ...] = MODELS,
    log=print,
) -> dict:
    pool = engine.pool
    lofar_bands = bands(pool.lofar_hz, LOFAR_BAND_HZ)
    demon_bands = bands(pool.demon_hz, DEMON_BAND_HZ)
    device = engine.device
    lofar_all = torch.as_tensor(pool.lofar, device=device)
    demon_all = torch.as_tensor(pool.demon, device=device)
    shapes = pool.lofar.shape[1:] + (pool.demon.shape[1],)
    set_determinism(SEED)

    results, importance_for_era = {}, {}
    for model_name in models:
        if model_name not in round3_summary["models"]:
            log(f"[{model_name}] not in the round 3 summary: skipped")
            continue
        entry = round3_summary["models"][model_name]
        config_name = entry["chosen_config"]
        config = entry["chosen_hyperparameters"]
        seeds = (
            [0] if family(model_name) == "baseline" else list(round3_summary["seeds"])
        )
        per_partition = []
        for partition in engine.partitions:
            rows = engine.rows[partition.name]["val"]
            labels = pool.labels[rows]
            measures = []
            for seed in seeds:
                weights_path, _ = engine.paths(
                    model_name, "final", config_name, partition, seed
                )
                if family(model_name) == "baseline":
                    val = Rows(
                        pool.lofar[rows],
                        pool.demon[rows],
                        labels,
                        pool.vessel_group[rows],
                        pool.lofar_hz,
                    )
                    lofar_drop, demon_drop = permutation_importance(
                        linear.load_baseline(weights_path),
                        val,
                        lofar_bands,
                        demon_bands,
                    )
                    measures.append({"lofar": lofar_drop, "demon": demon_drop})
                    continue
                net = build_network(model_name, config, shapes)
                net.load_state_dict(
                    torch.load(weights_path, map_location="cpu", weights_only=True)
                )
                net.to(device).eval()
                idx = torch.as_tensor(rows, device=device)
                lofar, demon = lofar_all[idx], demon_all[idx]
                o_lofar, o_demon, o_lofar_class, _ = occlusion(
                    net, lofar, demon, labels, lofar_bands, demon_bands
                )
                s_lofar, s_demon = saliency(
                    net, lofar, demon, labels, lofar_bands, demon_bands
                )
                measure = {
                    "lofar": o_lofar,
                    "demon": o_demon,
                    "lofar_per_class": o_lofar_class,
                    "saliency_lofar": s_lofar,
                    "saliency_demon": s_demon,
                }
                if hasattr(net, "keep_attention"):
                    attention = class_token_attention(net, lofar, demon)
                    centres = token_centres_hz(
                        model_name, attention.shape[1] - 2, pool.lofar_hz
                    )
                    measure["attention_demon"] = attention[:, 1]
                    measure["attention_bands"] = np.array(
                        [
                            attention_by_band(layer[2:], centres, lofar_bands)
                            for layer in attention
                        ]
                    )
                measures.append(measure)
            per_partition.append(
                {
                    key: np.mean([m[key] for m in measures], axis=0)
                    for key in measures[0]
                }
            )
            log(
                f"[{model_name}] {partition.name}: {len(seeds)} seed(s) measured on val"
            )
        keys = per_partition[0].keys()
        aggregated = {key: stats([p[key] for p in per_partition]) for key in keys}
        results[model_name] = {
            "config": config_name,
            "seeds": seeds,
            "metric": (
                "drop in val macro F1 when a band is shuffled across segments"
                if family(model_name) == "baseline"
                else "drop in the probability of the true class when a band is set to the "
                "train mean (occlusion); saliency |d logit / d input| per band and class"
            ),
            "measures": aggregated,
            "top_lofar_bands": top_bands(aggregated["lofar"]["mean"], lofar_bands),
            "top_demon_bands": top_bands(aggregated["demon"]["mean"], demon_bands, 3),
        }
        if "attention_bands" in aggregated:
            last = aggregated["attention_bands"]["mean"][-1]
            results[model_name]["top_attention_bands"] = top_bands(last, lofar_bands)
            results[model_name]["attention_demon_mean"] = aggregated["attention_demon"][
                "mean"
            ]
            importance_for_era[f"{model_name}_attention"] = last
        importance_for_era[model_name] = aggregated["lofar"]["mean"]

    everyone = Rows(
        pool.lofar, pool.demon, pool.labels, pool.vessel_group, pool.lofar_hz
    )
    nobody = Rows(
        pool.lofar[:0],
        pool.demon[:0],
        pool.labels[:0],
        pool.vessel_group[:0],
        pool.lofar_hz,
    )
    era = era_check(everyone, nobody, lofar_bands, importance_for_era)
    era["note"] = (
        "p-values are not reliable: neighbouring bands are correlated and several models are "
        "tested without correction"
    )
    report = {
        "round": 3,
        "split": "val of every partition (test not used); the era check describes the mean spectra of every vessel, without any model",
        "device": str(engine.device),
        "hardware": engine.hardware,
        "classes": list(CLASSES),
        "lofar_bands": [{"lo_hz": b["lo"], "hi_hz": b["hi"]} for b in lofar_bands],
        "demon_bands": [{"lo_hz": b["lo"], "hi_hz": b["hi"]} for b in demon_bands],
        "partitions": len(engine.partitions),
        "aggregation": "seeds averaged within a partition; mean and std over partitions",
        "models": results,
        "era_check": era,
        "code_sha256": engine.code,
        "cv_splits_sha256": engine.splits_sha256,
        "round3_code_sha256": round3_summary["code_sha256"],
    }
    write_metrics(output_path, report)
    return report


def load_round3_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
