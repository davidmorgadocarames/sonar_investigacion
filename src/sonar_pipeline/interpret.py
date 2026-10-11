"""Interpretability (F3-T10, D-018), computed on val with the frozen models; test is not used.

- Baseline: permutation importance of frequency bands (40 Hz of LOFAR, 10 Hz of DEMON): the
  drop in val macro F1 when the columns of one band are shuffled across segments.
- CNNs: saliency |d logit of the true class / d input|, averaged per class, and occlusion of
  each band (set to the train mean): the drop in the probability of the true class.
- Era check (DT-005): with the train + val vessels, (a) "era lines": LOFAR bins with a line
  (> 3 dB above the background in the vessel's mean spectrum) in at least 75 % of the vessels of
  one era and at most 25 % of the other, including mains hum (50/60 Hz and harmonics); (b) how
  much each band differs between eras (absolute difference of the mean level), and whether the
  bands a model relies on are the ones that differ most (Spearman rank correlation, top-5
  overlap).
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from sonar_pipeline.dataset import CLASSES, ERAS, baseline_features, load_split
from sonar_pipeline.metrics import f1_scores, write_metrics

LOFAR_BAND_HZ = 40.0
DEMON_BAND_HZ = 10.0
REPEATS = 10
SEED = 0
LINE_DB = 3.0
ERA_PRESENT = 0.75
ERA_ABSENT = 0.25
TOP = 5
HUM_HZ = tuple(f * k for f in (50, 60) for k in range(1, 6))


def bands(hz: np.ndarray, width: float) -> list[dict]:
    """Consecutive bands of `width` Hz; the last one also takes the top edge bin."""
    top = hz[-1]
    edges = np.arange(0.0, top, width)
    result = []
    for lo in edges:
        hi = lo + width
        rows = np.flatnonzero((hz >= lo) & ((hz < hi) | ((hi >= top) & (hz <= top))))
        result.append({"lo": float(lo), "hi": float(min(hi, top)), "bins": rows})
    return result


def summary(band_list: list[dict], values: np.ndarray) -> list[dict]:
    return [
        {"lo_hz": b["lo"], "hi_hz": b["hi"], "value": float(v)}
        for b, v in zip(band_list, values, strict=True)
    ]


# --------------------------------------------------------------------------- baseline


def baseline_importance(weights: dict, val, lofar_bands, demon_bands, seed=SEED):
    """Mean and std of the macro-F1 drop per band, and the mean per-class F1 drop."""
    from sonar_pipeline.baseline import predict_proba

    x = baseline_features(val)
    n_lofar = val.lofar.shape[2]
    base_macro, base_class = f1_scores(
        val.labels, predict_proba(weights, x).argmax(axis=1)
    )
    rng = np.random.default_rng(seed)
    columns = [b["bins"] for b in lofar_bands] + [
        n_lofar + b["bins"] for b in demon_bands
    ]
    macro_mean, macro_std, per_class = [], [], []
    for cols in columns:
        drops, class_drops = [], []
        for _ in range(REPEATS):
            shuffled = x.copy()
            shuffled[:, cols] = x[rng.permutation(len(x))][:, cols]
            macro, by_class = f1_scores(
                val.labels, predict_proba(weights, shuffled).argmax(axis=1)
            )
            drops.append(base_macro - macro)
            class_drops.append(np.array(base_class) - np.array(by_class))
        macro_mean.append(np.mean(drops))
        macro_std.append(np.std(drops))
        per_class.append(np.mean(class_drops, axis=0))
    n = len(lofar_bands)
    return {
        "metric": "drop in val macro F1 when a band is shuffled across segments",
        "base_f1_macro": base_macro,
        "repeats": REPEATS,
        "lofar": np.array(macro_mean[:n]),
        "lofar_std": np.array(macro_std[:n]),
        "demon": np.array(macro_mean[n:]),
        "demon_std": np.array(macro_std[n:]),
        "lofar_per_class": np.array(per_class[:n]),  # (bands, classes)
        "demon_per_class": np.array(per_class[n:]),
    }


# --------------------------------------------------------------------------- CNN


def cnn_saliency(model, val, batch_size: int = 64):
    """Mean |d logit(true class) / d input| per class: LOFAR (classes, frames, bins), DEMON."""
    import torch

    lofar_sum = np.zeros((len(CLASSES), *val.lofar.shape[1:]))
    demon_sum = np.zeros((len(CLASSES), val.demon.shape[1]))
    counts = np.bincount(val.labels, minlength=len(CLASSES))
    model.eval()
    for start in range(0, len(val), batch_size):
        lofar = torch.as_tensor(val.lofar[start : start + batch_size]).requires_grad_(
            True
        )
        demon = torch.as_tensor(val.demon[start : start + batch_size]).requires_grad_(
            True
        )
        labels = torch.as_tensor(val.labels[start : start + batch_size])
        logits = model(lofar, demon)
        logits.gather(1, labels[:, None]).sum().backward()
        for k in range(len(CLASSES)):
            mask = (labels == k).numpy()
            lofar_sum[k] += lofar.grad.abs().numpy()[mask].sum(axis=0)
            demon_sum[k] += demon.grad.abs().numpy()[mask].sum(axis=0)
    safe = np.maximum(counts, 1)
    return lofar_sum / safe[:, None, None], demon_sum / safe[:, None]


def true_class_probability(
    model, lofar: np.ndarray, demon: np.ndarray, labels
) -> np.ndarray:
    import torch

    from sonar_pipeline.cnn import predict_proba

    probs = predict_proba(model, torch.as_tensor(lofar), torch.as_tensor(demon))
    return probs[np.arange(len(labels)), labels]


def cnn_occlusion(model, val, lofar_bands, demon_bands):
    """Drop of the true-class probability when a band is set to the train mean."""
    base = true_class_probability(model, val.lofar, val.demon, val.labels)
    lofar_fill, demon_fill = float(model.lofar_mean), float(model.demon_mean)

    def drops(branch: str, band_list):
        means, per_class = [], []
        for band in band_list:
            lofar, demon = val.lofar.copy(), val.demon.copy()
            if branch == "lofar":
                lofar[:, :, band["bins"]] = lofar_fill
            else:
                demon[:, band["bins"]] = demon_fill
            drop = base - true_class_probability(model, lofar, demon, val.labels)
            means.append(drop.mean())
            per_class.append(
                [
                    drop[val.labels == k].mean() if np.any(val.labels == k) else 0.0
                    for k in range(len(CLASSES))
                ]
            )
        return np.array(means), np.array(per_class)

    lofar, lofar_class = drops("lofar", lofar_bands)
    demon, demon_class = drops("demon", demon_bands)
    return {
        "metric": "drop in the probability of the true class when a band is set to the "
        "train mean",
        "lofar": lofar,
        "demon": demon,
        "lofar_per_class": lofar_class,
        "demon_per_class": demon_class,
    }


# --------------------------------------------------------------------------- era check


def era_check(train, val, lofar_bands, importances: dict) -> dict:
    """Era lines and whether the important LOFAR bands are the ones that differ by era."""
    era_of = {CLASSES.index(c): era for era, pair in ERAS.items() for c in pair}
    spectra = np.concatenate([train.lofar.mean(axis=1), val.lofar.mean(axis=1)])
    groups = np.concatenate([train.vessel_group, val.vessel_group])
    labels = np.concatenate([train.labels, val.labels])
    names = list(dict.fromkeys(groups.tolist()))
    vessel = np.array([spectra[groups == g].mean(axis=0) for g in names])
    vessel_era = np.array([era_of[int(labels[groups == g][0])] for g in names])
    hz = train.lofar_hz

    eras = list(ERAS)
    has_line = vessel > LINE_DB
    fraction = {e: has_line[vessel_era == e].mean(axis=0) for e in eras}
    era_a, era_b = eras
    era_line = ((fraction[era_a] >= ERA_PRESENT) & (fraction[era_b] <= ERA_ABSENT)) | (
        (fraction[era_b] >= ERA_PRESENT) & (fraction[era_a] <= ERA_ABSENT)
    )
    hum = []
    for f in HUM_HZ:
        if f <= hz[-1]:
            i = int(np.argmin(np.abs(hz - f)))
            hum.append(
                {"hz": f, **{f"fraction_{e}": float(fraction[e][i]) for e in eras}}
            )

    level = {e: vessel[vessel_era == e].mean(axis=0) for e in eras}
    difference = np.array(
        [
            abs(level[era_a][band["bins"]].mean() - level[era_b][band["bins"]].mean())
            for band in lofar_bands
        ]
    )
    top_difference = set(np.argsort(-difference)[:TOP].tolist())
    models = {}
    for name, values in importances.items():
        rho, p = spearmanr(values, difference)
        top_model = set(np.argsort(-values)[:TOP].tolist())
        models[name] = {
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "top_overlap": len(top_model & top_difference),
            "top_bands_hz": [lofar_bands[i]["lo"] for i in sorted(top_model)],
        }
    return {
        "vessels": {e: int(np.sum(vessel_era == e)) for e in eras},
        "line_db": LINE_DB,
        "era_line_rule": f">= {ERA_PRESENT:.0%} of one era and <= {ERA_ABSENT:.0%} of the other",
        "era_lines_hz": [float(h) for h in hz[era_line]],
        "hum": hum,
        "band_difference_db": difference,
        "top_difference_bands_hz": [
            lofar_bands[i]["lo"] for i in sorted(top_difference)
        ],
        "models": models,
    }


# --------------------------------------------------------------------------- run


def run_interpretability(
    processed_dir: Path,
    models_dir: Path,
    results_dir: Path,
    figures_dir: Path | None = None,
    lofar_band_hz: float = LOFAR_BAND_HZ,
    demon_band_hz: float = DEMON_BAND_HZ,
) -> dict:
    from sonar_pipeline.baseline import load_baseline
    from sonar_pipeline.cnn import load_cnn

    train = load_split(processed_dir, "train")
    val = load_split(processed_dir, "val")
    lofar_bands = bands(val.lofar_hz, lofar_band_hz)
    demon_bands = bands(val.demon_hz, demon_band_hz)

    baseline = baseline_importance(
        load_baseline(models_dir / "baseline.npz"), val, lofar_bands, demon_bands
    )
    cnns, maps = {}, {}
    for name in ("cnn", "cnn_aug"):
        model = load_cnn(models_dir / f"{name}.pt")
        lofar_map, demon_map = cnn_saliency(model, val)
        maps[f"{name}_saliency_lofar"] = lofar_map.astype(np.float32)
        maps[f"{name}_saliency_demon"] = demon_map.astype(np.float32)
        occlusion = cnn_occlusion(model, val, lofar_bands, demon_bands)
        saliency_band = np.array(
            [
                [lofar_map[k][:, b["bins"]].mean() for b in lofar_bands]
                for k in range(len(CLASSES))
            ]
        )
        cnns[name] = {
            "occlusion": occlusion,
            "saliency_lofar_band_per_class": saliency_band,
        }

    era = era_check(
        train,
        val,
        lofar_bands,
        {
            "baseline": baseline["lofar"],
            "cnn": cnns["cnn"]["occlusion"]["lofar"],
            "cnn_aug": cnns["cnn_aug"]["occlusion"]["lofar"],
        },
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    maps_path = results_dir / "interpretability_maps.npz"
    np.savez(maps_path, lofar_hz=val.lofar_hz, demon_hz=val.demon_hz, **maps)
    report = {
        "split": "val",
        "classes": list(CLASSES),
        "lofar_band_hz": lofar_band_hz,
        "demon_band_hz": demon_band_hz,
        "baseline": {
            "metric": baseline["metric"],
            "base_f1_macro": baseline["base_f1_macro"],
            "repeats": baseline["repeats"],
            "lofar": summary(lofar_bands, baseline["lofar"]),
            "lofar_std": baseline["lofar_std"],
            "demon": summary(demon_bands, baseline["demon"]),
            "lofar_per_class": baseline["lofar_per_class"],
            "demon_per_class": baseline["demon_per_class"],
        },
        "cnn": {
            name: {
                "occlusion_metric": entry["occlusion"]["metric"],
                "occlusion_lofar": summary(lofar_bands, entry["occlusion"]["lofar"]),
                "occlusion_demon": summary(demon_bands, entry["occlusion"]["demon"]),
                "occlusion_lofar_per_class": entry["occlusion"]["lofar_per_class"],
                "saliency_lofar_band_per_class": entry["saliency_lofar_band_per_class"],
            }
            for name, entry in cnns.items()
        },
        "era_check": era,
        "maps": maps_path.name,
    }
    write_metrics(results_dir / "interpretability.json", report)

    metrics_path = results_dir / "metrics.json"
    if (
        metrics_path.is_file()
    ):  # add the summary to metrics.json without touching the rest
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["interpretability"] = report
        metrics.pop("schema", None)
        write_metrics(metrics_path, metrics)

    if figures_dir is not None:
        from sonar_pipeline.figures import plot_interpretability

        report["figures"] = [str(plot_interpretability(report, figures_dir))]
    return report
