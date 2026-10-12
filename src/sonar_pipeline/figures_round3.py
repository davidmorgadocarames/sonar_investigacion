"""Figures of round 3 (F5-T08): drawn only from the versioned summaries in docs/results/.

round3.json (scores per partition, comparisons), round3_details.json (speed, curves,
generalisation, vessels, agreement, calibration, ROC) and round3_interpretability.json. The
titles do not name the round: the README presents these results only (F5-T11).
Each architecture has one hue; its version with augmentation is a lighter tone of it, and every
model is also named on the axis or in the legend, so identity never relies on colour alone.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from sonar_pipeline.dataset import CLASSES
from sonar_pipeline.figures import GRID, INK, MUTED, SEQUENTIAL, _save, _style

ORDER = (
    "baseline",
    "cnn",
    "cnn_aug",
    "hybrid",
    "hybrid_aug",
    "transformer",
    "transformer_aug",
)
NAMES = {
    "baseline": "Línea base",
    "cnn": "CNN",
    "cnn_aug": "CNN + aum.",
    "hybrid": "Híbrido",
    "hybrid_aug": "Híbrido + aum.",
    "transformer": "Transformer",
    "transformer_aug": "Transformer + aum.",
}
COLORS = {
    "baseline": "#6b6b6b",
    "cnn": "#256abf",
    "cnn_aug": "#8fb8ea",
    "hybrid": "#b8561b",
    "hybrid_aug": "#eba273",
    "transformer": "#2e8b57",
    "transformer_aug": "#93cfa9",
}
# Offsets (points) of the model names in the cost figure, so neighbours do not overlap.
LABEL_OFFSETS = {
    "cnn": (6, -2),
    "cnn_aug": (6, -12),
    "hybrid": (6, 8),
    "hybrid_aug": (-75, 8),
    "transformer": (6, -12),
    "transformer_aug": (6, 4),
}
SHORT_CLASSES = {
    "Cargo": "Cargo",
    "Passengership": "Pasaje",
    "Tanker": "Tanker",
    "Tug": "Tug",
}
ERA_TITLES = {
    "2016": "Passengership frente a Tanker (2016)",
    "2017": "Cargo frente a Tug (2017)",
}


def present(summary: dict) -> list[str]:
    return [m for m in ORDER if m in summary["models"]]


def comma(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _title(fig, text: str) -> None:
    fig.suptitle(text[:1].upper() + text[1:], color=INK, fontsize=11, x=0.01, ha="left")


def _ygrid(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _model_ticks(ax, models, rotation=30) -> None:
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([NAMES[m] for m in models], rotation=rotation, ha="right")


# --------------------------------------------------------------------------- 1. box plot


def plot_box(summary: dict, path: Path) -> Path:
    models = present(summary)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    panels = [("4 clases", None), *[(ERA_TITLES[e], e) for e in ("2016", "2017")]]
    rng = np.random.default_rng(0)
    for ax, (title, era) in zip(axes, panels, strict=True):
        for i, m in enumerate(models):
            rows = summary["models"][m]["per_partition"]
            if era is None:
                values = [r["test_f1_macro_segment"] for r in rows]
            else:
                values = [r["test_within_era_segment"][era] for r in rows]
                values = [v for v in values if v is not None]
            box = ax.boxplot(
                values,
                positions=[i],
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": INK},
            )
            box["boxes"][0].set(facecolor=COLORS[m], alpha=0.55, edgecolor=MUTED)
            for part in ("whiskers", "caps"):
                for line in box[part]:
                    line.set_color(MUTED)
            ax.scatter(
                i + rng.uniform(-0.18, 0.18, len(values)),
                values,
                s=9,
                color=INK,
                alpha=0.35,
                zorder=3,
                linewidths=0,
            )
            ax.scatter(
                [i],
                [np.mean(values)],
                marker="D",
                s=30,
                color="white",
                edgecolors=INK,
                zorder=4,
            )
        chance = (
            np.mean(
                [
                    summary["models"][m]["test"]["chance_f1_macro"]["mean"]
                    for m in models
                ]
            )
            if era is None
            else 0.5
        )
        ax.axhline(chance, color=MUTED, linestyle="--", linewidth=1)
        ax.text(
            -0.45,
            chance + 0.012,
            f"azar {'medido ' if era is None else '≈ '}{comma(chance)}",
            color=MUTED,
            fontsize=7,
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
            zorder=5,
        )
        ax.set_title(title, color=INK, fontsize=9)
        _model_ticks(ax, models)
        _ygrid(ax)
        _style(ax)
    axes[0].set_ylabel("F1 macro de test por segmento\n(un punto por partición)")
    _title(fig, "F1 de test en las 25 particiones (◇ = media)")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- 2. pairs


def corrected_interval(comparison: dict, level: float = 0.95) -> tuple[float, float]:
    """Interval of the mean difference with the Nadeau-Bengio corrected variance."""
    j, ratio = comparison["partitions"], comparison["test_train_ratio"]
    se = np.sqrt(1 / j + ratio) * comparison["std_difference"]
    half = stats.t.ppf(0.5 + level / 2, comparison["df"]) * se
    mean = comparison["mean_difference"]
    return mean - half, mean + half


def plot_pairs(summary: dict, path: Path) -> Path:
    rows = summary["comparisons"]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.4))
    for y, c in enumerate(reversed(rows)):
        lo, hi = corrected_interval(c)
        ax.plot([lo, hi], [y, y], color=COLORS[c["model"]], linewidth=2.2)
        ax.scatter(
            [c["mean_difference"]],
            [y],
            color=COLORS[c["model"]],
            edgecolors=INK,
            zorder=3,
            s=28,
        )
        ax.text(
            1.02,
            y,
            f"p {comma(c['p'], 3)} · Holm {comma(c['p_holm'], 2)}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=7,
            color=MUTED,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [f"{NAMES[c['model']]} − {NAMES[c['against']]}" for c in reversed(rows)],
        fontsize=8,
    )
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel(
        "Diferencia de F1 macro de test (media de 25 particiones, IC 95 % corregido)"
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    _title(fig, "comparaciones fijadas de antemano (Nadeau y Bengio, Holm)")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- 3. augmentation


def plot_augmentation(summary: dict, path: Path) -> Path:
    archs = [
        a
        for a in ("cnn", "hybrid", "transformer")
        if a in summary["models"] and f"{a}_aug" in summary["models"]
    ]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rng = np.random.default_rng(1)
    for i, a in enumerate(archs):
        plain = [
            r["test_f1_macro_segment"] for r in summary["models"][a]["per_partition"]
        ]
        aug = [
            r["test_f1_macro_segment"]
            for r in summary["models"][f"{a}_aug"]["per_partition"]
        ]
        d = np.array(aug) - np.array(plain)
        box = ax.boxplot(
            d,
            positions=[i],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": INK},
        )
        box["boxes"][0].set(facecolor=COLORS[f"{a}_aug"], alpha=0.6, edgecolor=MUTED)
        ax.scatter(
            i + rng.uniform(-0.15, 0.15, len(d)),
            d,
            s=10,
            color=INK,
            alpha=0.4,
            linewidths=0,
            zorder=3,
        )
        ax.text(
            i,
            d.max() + 0.02,
            f"mejora en {int((d > 0).sum())}/{len(d)}",
            ha="center",
            fontsize=7,
            color=MUTED,
        )
    ax.axhline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels([NAMES[a] for a in archs])
    ax.set_ylabel("F1 con aumento − F1 sin aumento\n(por partición)")
    _ygrid(ax)
    _style(ax)
    _title(fig, "efecto del aumento de datos")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- 4-5. cost, speed


def _f1(summary, m):
    t = summary["models"][m]["test"]["f1_macro_segment"]
    return t["mean"], t["std"]


def plot_cost(summary: dict, details: dict, path: Path) -> Path:
    models = present(summary)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    xs = [
        (
            "MFLOPs por segmento (inferencia)",
            lambda m: summary["models"][m]["cost"]["mflops_per_segment"],
        ),
        ("Parámetros", lambda m: summary["models"][m]["cost"]["parameters"]),
        (
            "Segundos por entrenamiento (media)",
            lambda m: details["models"][m]["speed"]["training_seconds_mean"],
        ),
    ]
    for ax, (label, getter) in zip(axes, xs, strict=True):
        for m in models:
            x = getter(m)
            mean, std = _f1(summary, m)
            ax.errorbar(
                [x],
                [mean],
                yerr=[std],
                fmt="o",
                color=COLORS[m],
                ecolor=COLORS[m],
                markeredgecolor=INK,
                capsize=3,
                markersize=6,
            )
            ax.annotate(
                NAMES[m],
                (x, mean),
                textcoords="offset points",
                xytext=LABEL_OFFSETS.get(m, (5, 4)),
                fontsize=7,
                color=INK,
            )
        ax.set_xscale("log")
        ax.set_xlabel(label)
        _ygrid(ax)
        _style(ax)
    axes[0].set_ylabel("F1 macro de test (media ± desviación)")
    _title(fig, "precisión frente a coste")
    fig.tight_layout()
    return _save(fig, path)


def plot_speed(details: dict, path: Path) -> Path:
    models = [m for m in ORDER if m in details["models"]]
    speed = {m: details["models"][m]["speed"] for m in models}
    panels = [
        (
            "Latencia de 1 segmento (ms)",
            lambda s: (
                1000 * s["inference_latency_seconds"]
                if s["inference_latency_seconds"]
                else None
            ),
        ),
        ("Segmentos/s en entrenamiento", lambda s: s["training_segments_per_second"]),
        ("Tokens/s en entrenamiento", lambda s: s["training_tokens_per_second"]),
        (
            "Formar minibatches (% del entrenamiento)",
            lambda s: (
                100 * s["minibatch_share_of_training"]
                if s.get("minibatch_share_of_training") is not None
                else None
            ),
        ),
        (
            "Memoria máxima de GPU (MiB)",
            lambda s: (
                s["gpu_peak_allocated_bytes_max"] / 2**20
                if s["gpu_peak_allocated_bytes_max"]
                else None
            ),
        ),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.8 * len(panels), 4))
    for ax, (label, getter) in zip(axes, panels, strict=True):
        for i, m in enumerate(models):
            value = getter(speed[m])
            if value is None:
                ax.text(i, 0, "n/a", ha="center", va="bottom", fontsize=7, color=MUTED)
                continue
            ax.bar(i, value, color=COLORS[m], edgecolor="white", width=0.7)
            ax.text(
                i,
                value,
                f"{value:,.0f}".replace(",", ".") if value >= 10 else comma(value, 2),
                ha="center",
                va="bottom",
                fontsize=7,
                color=INK,
            )
        ax.set_title(label, color=INK, fontsize=9)
        _model_ticks(ax, models, rotation=45)
        _ygrid(ax)
        _style(ax)
    _title(fig, "velocidad y memoria (RTX 4060; la línea base, en CPU)")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- 6-9. training


def plot_curves(details: dict, path: Path, min_share: float = 0.1) -> Path:
    models = [
        m for m in ORDER if m in details["models"] and details["models"][m]["curves"]
    ]
    keys = [
        ("train_loss", "Pérdida de entrenamiento"),
        ("val_loss", "Pérdida de validación"),
        ("val_f1_macro", "F1 macro de validación"),
    ]
    if models and all(
        "grad_norm_mean" in details["models"][m]["curves"] for m in models
    ):
        keys.append(("grad_norm_mean", "Norma del gradiente (media por época)"))
    fig, axes = plt.subplots(1, len(keys), figsize=(4.7 * len(keys), 4.2))
    for ax, (key, label) in zip(axes, keys, strict=True):
        for m in models:
            c = details["models"][m]["curves"]
            runs = np.array(c["runs"])
            keep = runs >= min_share * runs[0]
            epochs = np.array(c["epoch"])[keep]
            mean = np.array(c[key]["mean"])[keep]
            std = np.array(c[key]["std"])[keep]
            style = "--" if m.endswith("_aug") else "-"
            ax.plot(epochs, mean, style, color=COLORS[m], linewidth=1.6, label=NAMES[m])
            ax.fill_between(
                epochs, mean - std, mean + std, color=COLORS[m], alpha=0.12, linewidth=0
            )
        ax.set_title(label, color=INK, fontsize=9)
        ax.set_xlabel("Época")
        _ygrid(ax)
        _style(ax)
    axes[2].legend(fontsize=7, frameon=False)
    if len(keys) == 4:
        axes[3].set_yscale("log")
    _title(
        fig,
        f"curvas de aprendizaje medias de los entrenamientos finales "
        f"(mientras siguen ≥ {min_share:.0%} de ellos)",
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_generalisation(details: dict, path: Path) -> Path:
    models = [m for m in ORDER if m in details["models"]]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    roles = [
        ("train_f1_macro", "Entrenamiento", "o"),
        ("val_f1_macro", "Validación", "s"),
        ("test_f1_macro", "Test", "D"),
    ]
    for j, (key, label, marker) in enumerate(roles):
        for i, m in enumerate(models):
            values = [g[key] for g in details["models"][m]["generalisation"]]
            ax.errorbar(
                [i + (j - 1) * 0.22],
                [np.mean(values)],
                yerr=[np.std(values)],
                fmt=marker,
                color=COLORS[m],
                markeredgecolor=INK,
                capsize=2,
                label=label if i == 0 else None,
            )
    ax.set_ylabel("F1 macro (media ± desv., 25 particiones)")
    _model_ticks(ax, models)
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=INK,
            label=label,
        )
        for _, label, marker in roles
    ]
    ax.legend(
        handles=handles,
        fontsize=8,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.72),
    )
    _ygrid(ax)
    _style(ax)
    _title(fig, "brecha de generalización (conjunto de semillas)")
    fig.tight_layout()
    return _save(fig, path)


def plot_variance(details: dict, path: Path) -> Path:
    models = [
        m
        for m in ORDER
        if m in details["models"]
        and details["models"][m]["variance"]["seed_std"] is not None
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.38
    for i, m in enumerate(models):
        v = details["models"][m]["variance"]
        ax.bar(
            i - width / 2, v["partition_std"], width, color=COLORS[m], edgecolor="white"
        )
        ax.bar(
            i + width / 2,
            v["seed_std"],
            width,
            color=COLORS[m],
            edgecolor=INK,
            hatch="///",
            alpha=0.6,
        )
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(
                facecolor="white",
                edgecolor=INK,
                label="entre particiones (media de semillas)",
            ),
            Patch(
                facecolor="white",
                edgecolor=INK,
                hatch="///",
                label="entre semillas (misma partición)",
            ),
        ],
        fontsize=8,
        frameon=False,
    )
    ax.set_ylabel("Desviación del F1 macro de test")
    _model_ticks(ax, models)
    _ygrid(ax)
    _style(ax)
    _title(fig, "de dónde viene la variación: partición frente a semilla")
    fig.tight_layout()
    return _save(fig, path)


def plot_selection(summary: dict, path: Path) -> Path:
    families = [
        ("baseline", ("baseline",)),
        ("cnn", ("cnn", "cnn_aug")),
        ("hybrid", ("hybrid", "hybrid_aug")),
        ("transformer", ("transformer", "transformer_aug")),
    ]
    families = [(f, [m for m in ms if m in summary["models"]]) for f, ms in families]
    families = [(f, ms) for f, ms in families if ms]
    fig, axes = plt.subplots(
        1, len(families), figsize=(4 * len(families), 4.2), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, (fam, ms) in zip(axes, families, strict=True):
        configs = list(summary["models"][ms[0]]["selection"])
        for k, m in enumerate(ms):
            entry = summary["models"][m]
            offset = (k - (len(ms) - 1) / 2) * 0.25
            for i, c in enumerate(configs):
                s = entry["selection"][c]
                chosen = c == entry["chosen_config"]
                ax.errorbar(
                    [i + offset],
                    [s["val_f1_macro_mean"]],
                    yerr=[s["val_f1_macro_std"]],
                    fmt="o",
                    color=COLORS[m],
                    markeredgecolor=INK,
                    capsize=2,
                    markerfacecolor=COLORS[m] if chosen else "white",
                )
            test = entry["test"]["f1_macro_segment"]["mean"]
            ax.scatter(
                [configs.index(entry["chosen_config"]) + offset],
                [test],
                marker="D",
                color=COLORS[m],
                edgecolors=INK,
                zorder=4,
                s=30,
            )
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs)
        ax.set_xlabel("Configuración")
        ax.set_title(
            NAMES[ms[0]] if fam != "baseline" else "Línea base (C de 0,001 a 10)",
            color=INK,
            fontsize=9,
        )
        from matplotlib.lines import Line2D

        ax.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker="o",
                    color=COLORS[m],
                    markeredgecolor=INK,
                    label=NAMES[m],
                )
                for m in ms
            ],
            fontsize=7,
            frameon=False,
        )
        _ygrid(ax)
        _style(ax)
    axes[0].set_ylabel("F1 macro de validación (media ± desv.)\n◇ = test de la elegida")
    _title(fig, "elección de configuración con validación (relleno = elegida)")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- 10-15. results


def plot_confusions(summary: dict, path: Path) -> Path:
    models = present(summary)
    cols = 4
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, models, strict=False):
        test = summary["models"][m]["test"]
        normalised = np.array(test["joint_confusion_normalized"])
        counts = np.array(test["joint_confusion"])
        ax.imshow(normalised, cmap=SEQUENTIAL, vmin=0, vmax=1)
        for r in range(len(CLASSES)):
            for c in range(len(CLASSES)):
                ax.text(
                    c,
                    r,
                    f"{comma(normalised[r, c])}\n({counts[r, c]})",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if normalised[r, c] > 0.55 else INK,
                )
        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels([SHORT_CLASSES[c] for c in CLASSES], fontsize=7)
        ax.set_yticklabels([SHORT_CLASSES[c] for c in CLASSES], fontsize=7)
        ax.set_title(NAMES[m], color=INK, fontsize=9)
    for ax in axes[len(models) :]:
        ax.axis("off")
    fig.supxlabel("Clase predicha", color=INK, fontsize=9)
    fig.supylabel("Clase real", color=INK, fontsize=9)
    _title(
        fig, "matrices de confusión conjuntas de test (25 particiones, por segmento)"
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_vessels(details: dict, path: Path) -> Path:
    models = [m for m in ORDER if m in details["models"]]
    vessels = sorted(
        details["models"][models[0]]["vessels"], key=lambda g: (g.split("/")[0], g)
    )
    values = np.array(
        [
            [details["models"][m]["vessels"][g]["accuracy"] for m in models]
            for g in vessels
        ]
    )
    fig, ax = plt.subplots(figsize=(7.5, 0.2 * len(vessels) + 1.8))
    image = ax.imshow(values, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([NAMES[m] for m in models], rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(vessels)))
    ax.set_yticklabels([g.replace("/", " · ") for g in vessels], fontsize=6)
    fig.colorbar(
        image, ax=ax, fraction=0.04, label="Fracción de grabaciones de test acertadas"
    )
    _title(fig, "acierto por buque (grabación, todas sus veces en test)")
    fig.tight_layout()
    return _save(fig, path)


def plot_agreement(details: dict, path: Path) -> Path:
    models = [m for m in ORDER if m in details["models"]]
    n = len(models)
    same, wrong = np.full((n, n), np.nan), np.full((n, n), np.nan)
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            entry = details["agreement_recording_level"].get(f"{a}|{b}") or details[
                "agreement_recording_level"
            ].get(f"{b}|{a}")
            if entry:
                same[i, j] = entry["same_prediction"]
                wrong[i, j] = entry["both_wrong"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, matrix, label in zip(
        axes, (same, wrong), ("Misma predicción", "Ambos fallan"), strict=True
    ):
        ax.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=1)
        for i in range(n):
            for j in range(n):
                if not np.isnan(matrix[i, j]):
                    ax.text(
                        j,
                        i,
                        comma(matrix[i, j]),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if matrix[i, j] > 0.55 else INK,
                    )
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(
            [NAMES[m] for m in models], rotation=40, ha="right", fontsize=7
        )
        ax.set_yticklabels([NAMES[m] for m in models], fontsize=7)
        ax.set_title(
            label + " (fracción de grabaciones de test)", color=INK, fontsize=9
        )
    _title(fig, "acuerdo entre modelos por grabación")
    fig.tight_layout()
    return _save(fig, path)


def plot_calibration(details: dict, path: Path) -> Path:
    models = [m for m in ORDER if m in details["models"]]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot(
        [0, 1], [0, 1], "--", color=MUTED, linewidth=1, label="calibración perfecta"
    )
    for m in models:
        cal = details["models"][m]["calibration"]
        conf = [b["confidence"] for b in cal["bins"]]
        acc = [b["accuracy"] for b in cal["bins"]]
        ax.plot(
            conf,
            acc,
            color=COLORS[m],
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            linestyle="--" if m.endswith("_aug") else "-",
            label=f"{NAMES[m]} (ECE {comma(cal['ece'])}, Brier {comma(cal['brier'])})",
        )
    ax.set_xlabel("Confianza (probabilidad de la clase predicha)")
    ax.set_ylabel("Acierto real")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    _style(ax)
    _title(fig, "calibración (test, segmentos de las 25 particiones)")
    fig.tight_layout()
    return _save(fig, path)


def plot_roc(details: dict, path: Path) -> Path:
    models = [m for m in ORDER if m in details["models"]]
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(15, 4), sharey=True)
    for ax, cls in zip(axes, CLASSES, strict=True):
        ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1)
        for m in models:
            roc = details["models"][m]["roc"][cls]
            ax.plot(
                roc["pfa"],
                roc["pd"],
                color=COLORS[m],
                linewidth=1.3,
                linestyle="--" if m.endswith("_aug") else "-",
                label=f"{NAMES[m]} {comma(roc['auc'])}",
            )
        ax.set_title(f"{cls} frente al resto (AUC)", color=INK, fontsize=9)
        ax.set_xlabel("Pfa")
        ax.legend(fontsize=6, frameon=False, loc="lower right")
        _style(ax)
    axes[0].set_ylabel("Pd")
    _title(
        fig, "ROC de clasificación conjuntas (test, segmentos de las 25 particiones)"
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_interpretability(report: dict, path: Path) -> Path:
    models = [m for m in ORDER if m in report["models"]]
    lofar_bands = report["lofar_bands"]
    rows, labels = [], []
    for m in models:
        values = np.array(report["models"][m]["measures"]["lofar"]["mean"])
        rows.append(values / max(values.max(), 1e-12))
        labels.append(NAMES[m])
    difference = np.array(report["era_check"]["band_difference_db"])
    rows.append(difference / difference.max())
    labels.append("Diferencia entre épocas")
    attention = [
        m for m in models if "attention_bands" in report["models"][m]["measures"]
    ]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 6.5),
        gridspec_kw={"height_ratios": [len(rows), max(len(attention), 1)]},
    )
    image = axes[0].imshow(
        np.clip(rows, 0, 1), cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto"
    )
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].axhline(len(models) - 0.5, color=INK, linewidth=1)
    axes[0].set_title(
        "Importancia por banda de 40 Hz (normalizada al máximo de cada fila; "
        "línea base: permutación; redes: oclusión)",
        color=INK,
        fontsize=9,
    )
    att = [
        np.array(report["models"][m]["measures"]["attention_bands"]["mean"])[-1]
        for m in attention
    ]
    if att:
        axes[1].imshow(np.array(att), cmap=SEQUENTIAL, aspect="auto")
        axes[1].set_yticks(range(len(attention)))
        axes[1].set_yticklabels([NAMES[m] for m in attention], fontsize=7)
    axes[1].set_title(
        "Atención del token de clase por banda (última capa)", color=INK, fontsize=9
    )
    ticks = list(range(0, len(lofar_bands), 5))
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{lofar_bands[i]['lo_hz']:g}" for i in ticks], fontsize=7)
    axes[1].set_xlabel("Inicio de la banda (Hz)")
    fig.colorbar(image, ax=axes, fraction=0.02)
    _title(fig, "qué frecuencias usa cada modelo (validación)")
    return _save(fig, path)


FIGURES = {
    "box": "round3_box.png",
    "pairs": "round3_pairs.png",
    "augmentation": "round3_augmentation.png",
    "cost": "round3_cost.png",
    "speed": "round3_speed.png",
    "curves": "round3_curves.png",
    "generalisation": "round3_generalisation.png",
    "variance": "round3_variance.png",
    "selection": "round3_selection.png",
    "confusion": "round3_confusion.png",
    "vessels": "round3_vessels.png",
    "agreement": "round3_agreement.png",
    "calibration": "round3_calibration.png",
    "interpretability": "round3_interpretability.png",
    "roc": "round3_roc.png",
}


def plot_round3(
    summary: dict, details: dict, interpretability: dict, folder: Path
) -> dict:
    """All 15 figures; returns {key: path}."""
    f = {k: folder / v for k, v in FIGURES.items()}
    plot_box(summary, f["box"])
    plot_pairs(summary, f["pairs"])
    plot_augmentation(summary, f["augmentation"])
    plot_cost(summary, details, f["cost"])
    plot_speed(details, f["speed"])
    plot_curves(details, f["curves"])
    plot_generalisation(details, f["generalisation"])
    plot_variance(details, f["variance"])
    plot_selection(summary, f["selection"])
    plot_confusions(summary, f["confusion"])
    plot_vessels(details, f["vessels"])
    plot_agreement(details, f["agreement"])
    plot_calibration(details, f["calibration"])
    plot_interpretability(interpretability, f["interpretability"])
    plot_roc(details, f["roc"])
    return f
