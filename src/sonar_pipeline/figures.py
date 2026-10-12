"""Figures (D-016): LOFARgram and DEMON of a recording, and spectra with a known frequency marked.

PNG files without a display (Agg backend). One-hue sequential blue for magnitude, one thin
line per spectrum, text in neutral ink. Figure text is in Spanish, like the documentation.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from sonar_pipeline.params import (
    DEMON_MIN_HZ,
    LOFAR_HOP_S,
    LOFAR_WINDOW_S,
    SEGMENT_S,
)

INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#e4e4e4"
LINE = "#256abf"
# Light -> dark blue, from near the surface (0 dB, background) to the strongest lines. Gaps
# between segments (NaN rows) show the surface.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "sequential_blue",
    [
        "#f7fbff",
        "#cde2fb",
        "#9ec5f4",
        "#6da7ec",
        "#3987e5",
        "#256abf",
        "#184f95",
        "#0d366b",
    ],
).with_extremes(bad="white")
LOFAR_DB_RANGE = (0.0, 15.0)
DPI = 150


def _style(ax):
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return path


def lofar_time_grid(
    lofar_segments: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Place the frames of consecutive segments on a uniform grid of LOFAR_HOP_S.

    Frame k of segment i is centred at i * SEGMENT_S + LOFAR_WINDOW_S / 2 + k * LOFAR_HOP_S.
    Grid rows without a frame (between segments) are NaN. Returns the rows and the time of the
    top and bottom edges (s).
    """
    n_segments, n_frames, n_bins = lofar_segments.shape
    steps_per_segment = SEGMENT_S / LOFAR_HOP_S
    if steps_per_segment != round(steps_per_segment):
        raise ValueError(
            "SEGMENT_S must be a multiple of LOFAR_HOP_S to draw the LOFARgram"
        )
    steps_per_segment = round(steps_per_segment)
    n_rows = (n_segments - 1) * steps_per_segment + n_frames
    rows = np.full((n_rows, n_bins), np.nan)
    for i in range(n_segments):
        start = i * steps_per_segment
        rows[start : start + n_frames] = lofar_segments[i]
    first_centre = LOFAR_WINDOW_S / 2
    top = first_centre - LOFAR_HOP_S / 2
    return rows, (top, top + n_rows * LOFAR_HOP_S)


def plot_recording(
    lofar_segments: np.ndarray,
    lofar_hz: np.ndarray,
    demon_spectrum: np.ndarray,
    demon_hz: np.ndarray,
    title: str,
    path: Path,
) -> Path:
    """LOFARgram of consecutive segments (time downwards) above the recording's DEMON.

    lofar_segments has shape (n_segments, n_frames, n_bins). Every frame is drawn as one row of
    LOFAR_HOP_S at its true time (frame centre), and the gap between segments stays empty.
    """
    rows, (time_top, time_bottom) = lofar_time_grid(lofar_segments)
    bin_hz = lofar_hz[1] - lofar_hz[0]

    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1.3]},
        layout="constrained",
    )
    fig.suptitle(title, color=INK, fontsize=11, x=0.02, ha="left")

    image = top.imshow(
        rows,
        aspect="auto",
        origin="upper",
        cmap=SEQUENTIAL,
        vmin=LOFAR_DB_RANGE[0],
        vmax=LOFAR_DB_RANGE[1],
        extent=(
            lofar_hz[0] - bin_hz / 2,
            lofar_hz[-1] + bin_hz / 2,
            time_bottom,
            time_top,
        ),
        interpolation="nearest",
    )
    top.set_title("LOFARgram", color=INK, fontsize=9, loc="left")
    top.set_xlabel("Frecuencia (Hz)")
    top.set_ylabel("Tiempo (s)")
    _style(top)
    bar = fig.colorbar(image, ax=top, pad=0.01)
    bar.set_label("dB sobre el fondo", color=INK, fontsize=8)
    bar.ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    bar.outline.set_visible(False)

    # Bins below DEMON_MIN_HZ are set to 0 by design (D-015): they are not drawn.
    kept = demon_hz >= DEMON_MIN_HZ
    bottom.plot(demon_hz[kept], demon_spectrum[kept], color=LINE, linewidth=1.2)
    bottom.set_title(
        "DEMON (media de los segmentos)", color=INK, fontsize=9, loc="left"
    )
    bottom.set_xlabel("Frecuencia de modulación (Hz)")
    bottom.set_ylabel("Amplitud relativa")
    bottom.set_xlim(0, demon_hz[-1])
    bottom.set_ylim(0, 1.05)
    bottom.grid(color=GRID, linewidth=0.6)
    bottom.set_axisbelow(True)
    _style(bottom)
    return _save(fig, path)


def plot_spectrum_with_marker(
    hz: np.ndarray,
    values: np.ndarray,
    marker_hz: float,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> Path:
    """One spectrum with a dashed vertical line at the frequency that should peak."""
    fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
    ax.plot(hz, values, color=LINE, linewidth=1.2)
    ax.axvline(marker_hz, color=MUTED, linewidth=1.0, linestyle="--")
    top = np.max(values)
    ax.annotate(
        f"frecuencia conocida: {marker_hz:g} Hz",
        xy=(marker_hz, top),
        xytext=(6, -4),
        textcoords="offset points",
        color=INK,
        fontsize=8,
        va="top",
    )
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(hz[0], hz[-1])
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    _style(ax)
    return _save(fig, path)


# --------------------------------------------------------------------------- results (F3-T09)

# Fixed categorical order: a model keeps its colour in every figure.
MODEL_COLOURS = {"baseline": "#2a78d6", "cnn": "#eb6834", "cnn_aug": "#1baf7a"}
MODEL_LABELS = {
    "baseline": "Línea base",
    "cnn": "CNN sin aumento",
    "cnn_aug": "CNN con aumento",
}
LEVEL_LABELS = {"segment": "por segmento", "recording": "por grabación"}
CLASS_SHORT = {
    "Cargo": "Cargo",
    "Passengership": "Passeng.",
    "Tanker": "Tanker",
    "Tug": "Tug",
}


def plot_confusions(models: dict, path: Path) -> Path:
    """Test confusion matrices normalised by true class: one row per level, one column per
    model."""
    names = list(models)
    classes = list(CLASS_SHORT)
    fig, axes = plt.subplots(
        2,
        len(names),
        figsize=(3.4 * len(names), 6.6),
        layout="constrained",
        squeeze=False,
    )
    fig.suptitle(
        "Test: matrices de confusión normalizadas por clase real",
        color=INK,
        fontsize=11,
        x=0.02,
        ha="left",
    )
    for row, level in enumerate(("segment", "recording")):
        for col, name in enumerate(names):
            ax = axes[row][col]
            matrix = np.array(models[name]["test"][level]["confusion_normalized"])
            counts = np.array(models[name]["test"][level]["confusion"])
            image = ax.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=1)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    colour = "white" if matrix[i, j] > 0.6 else INK
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.2f}\n({counts[i, j]})",
                        ha="center",
                        va="center",
                        color=colour,
                        fontsize=7,
                    )
            short = [CLASS_SHORT[c] for c in classes]
            ax.set_xticks(range(len(classes)), short, fontsize=7)
            ax.set_yticks(range(len(classes)), short, fontsize=7)
            ax.set_title(
                f"{MODEL_LABELS[name]}, {LEVEL_LABELS[level]}",
                color=INK,
                fontsize=8,
                loc="left",
            )
            if row == 1:
                ax.set_xlabel("Predicha", color=INK, fontsize=8)
            if col == 0:
                ax.set_ylabel("Real", color=INK, fontsize=8)
            ax.tick_params(colors=MUTED, labelcolor=INK)
    bar = fig.colorbar(image, ax=axes, shrink=0.6, pad=0.01)
    bar.set_label("Fracción de la clase real", color=INK, fontsize=8)
    bar.outline.set_visible(False)
    return _save(fig, path)


def _f1_point(scores: dict | None) -> tuple[float, float, float] | None:
    if scores is None:
        return None
    lo, hi = scores["ci95"]["f1_macro"]
    return scores["f1_macro"], lo, hi


def plot_f1_intervals(models: dict, path: Path) -> Path:
    """Test macro F1 with its 95 % bootstrap interval: 4 classes and within each era."""
    groups = [
        ("4 clases", None),
        ("2016: Passeng. vs Tanker", "2016"),
        ("2017: Cargo vs Tug", "2017"),
    ]
    names = list(models)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained", sharey=True)
    fig.suptitle(
        "Test: F1 macro con intervalo del 95 % (bootstrap por grabaciones)",
        color=INK,
        fontsize=11,
        x=0.02,
        ha="left",
    )
    offsets = np.linspace(-0.22, 0.22, len(names))
    for ax, level in zip(axes, ("segment", "recording"), strict=True):
        for g, (_, era) in enumerate(groups):
            first = models[names[0]]["test"]
            reference = first[level] if era is None else first["within_era"][era][level]
            # Measured chance (random classifier on the true labels), same for every model;
            # no line when the scores are missing (an era without both classes).
            if reference is not None:
                chance = reference["chance_f1_macro"]
                ax.plot(
                    [g - 0.35, g + 0.35],
                    [chance, chance],
                    color=MUTED,
                    lw=1,
                    ls="--",
                    label="azar (medido)" if g == 0 else None,
                )
            for offset, name in zip(offsets, names, strict=True):
                test = models[name]["test"]
                scores = test[level] if era is None else test["within_era"][era][level]
                point = _f1_point(scores)
                if point is None:
                    continue
                value, lo, hi = point
                ax.errorbar(
                    g + offset,
                    value,
                    yerr=[[value - lo], [hi - value]],
                    fmt="o",
                    color=MODEL_COLOURS[name],
                    markersize=7,
                    elinewidth=2,
                    capsize=0,
                    label=MODEL_LABELS[name] if g == 0 else None,
                )
        ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"F1 macro {LEVEL_LABELS[level]}", color=INK, fontsize=9, loc="left"
        )
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        _style(ax)
    axes[0].set_ylabel("F1 macro")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, path)


def plot_roc(models: dict, path: Path, level: str = "segment") -> Path:
    """Test ROC per class (one against the rest): Pd against Pfa on a logarithmic axis."""
    classes = list(CLASS_SHORT)
    fig, axes = plt.subplots(
        1, len(classes), figsize=(13, 3.6), layout="constrained", sharey=True
    )
    fig.suptitle(
        f"Test: ROC de clasificación {LEVEL_LABELS[level]}, una clase frente al resto "
        "(no de detección: sin grabaciones de solo fondo)",
        color=INK,
        fontsize=11,
        x=0.02,
        ha="left",
    )
    floor = 1e-3
    for ax, cls in zip(axes, classes, strict=True):
        for name, entry in models.items():
            roc = entry["test"][level]["roc"][cls]
            if roc is None:
                continue
            pfa = np.maximum(np.array(roc["pfa"]), floor)
            ax.step(
                pfa,
                roc["pd"],
                where="post",
                color=MODEL_COLOURS[name],
                linewidth=1.5,
                label=f"{MODEL_LABELS[name]} (AUC {roc['auc']:.2f})",
            )
        ax.axvline(0.01, color=MUTED, linewidth=1, linestyle=":")
        rocs = [e["test"][level]["roc"][cls] for e in models.values()]
        negatives = next((r["n_negatives"] for r in rocs if r), 0)
        if (
            0 < negatives < 100
        ):  # Pd at Pfa = 1 % is null in metrics.json (D-018 adj. 2)
            ax.text(
                0.011,
                0.5,
                f"Pfa = 1 % no medible ({negatives} negativos)",
                color=MUTED,
                fontsize=6.5,
                rotation=90,
                va="center",
            )
        ax.set_xscale("log")
        ax.set_xlim(floor, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(cls, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("Pfa (escala log.)")
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=6.5, loc="upper left")
        _style(ax)
    axes[0].set_ylabel("Pd")
    return _save(fig, path)


def plot_results(models: dict, figures_dir: Path) -> list[Path]:
    return [
        plot_confusions(models, figures_dir / "results_confusion.png"),
        plot_f1_intervals(models, figures_dir / "results_f1.png"),
        plot_roc(models, figures_dir / "results_roc_segment.png", "segment"),
        plot_roc(models, figures_dir / "results_roc_recording.png", "recording"),
    ]


def plot_interpretability(report: dict, figures_dir: Path) -> Path:
    """Per LOFAR band (val): baseline permutation importance, occlusion of both CNNs, and how
    much the band differs between recording eras (DT-005). One panel per measure."""
    lows = np.array([b["lo_hz"] for b in report["baseline"]["lofar"]])
    width = report["lofar_band_hz"] * 0.85
    panels = [
        (
            "Línea base: caída del F1 macro al permutar la banda",
            [b["value"] for b in report["baseline"]["lofar"]],
            MODEL_COLOURS["baseline"],
        ),
        (
            "CNN sin aumento: caída de la probabilidad de la clase real al ocultar la banda",
            [b["value"] for b in report["cnn"]["cnn"]["occlusion_lofar"]],
            MODEL_COLOURS["cnn"],
        ),
        (
            "CNN con aumento: caída de la probabilidad de la clase real al ocultar la banda",
            [b["value"] for b in report["cnn"]["cnn_aug"]["occlusion_lofar"]],
            MODEL_COLOURS["cnn_aug"],
        ),
        (
            "Diferencia de nivel medio entre épocas (dB, buques de train + val)",
            report["era_check"]["band_difference_db"],
            MUTED,
        ),
    ]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(10, 9), sharex=True, layout="constrained"
    )
    fig.suptitle(
        f"Interpretabilidad en val por bandas de LOFAR de {report['lofar_band_hz']:g} Hz",
        color=INK,
        fontsize=11,
        x=0.02,
        ha="left",
    )
    for ax, (title, values, colour) in zip(axes, panels, strict=True):
        ax.bar(lows + report["lofar_band_hz"] / 2, values, width=width, color=colour)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(title, color=INK, fontsize=8.5, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        _style(ax)
    axes[-1].set_xlabel("Frecuencia (Hz)")
    axes[-1].set_xlim(0, lows[-1] + report["lofar_band_hz"])
    return _save(fig, figures_dir / "interpretability_bands.png")


# --------------------------------------------------------------------------- data (F4-T02)

SPLIT_ORDER = ("train", "val", "test")
SPLIT_LABELS = {"train": "Entrenamiento", "val": "Validación", "test": "Test"}
# Splits are categories of their own: slots 4-6 of the palette, so they are not mistaken for the
# model colours (slots 1-3) in the result figures.
SPLIT_COLOURS = {"train": "#eda100", "val": "#e87ba4", "test": "#4a3aa7"}


def data_minutes(vessels_csv: Path, split_csv: Path) -> dict:
    """Minutes of audio, recordings and vessels per class and split (no audio needed)."""
    import csv

    with vessels_csv.open(newline="", encoding="utf-8") as f:
        seconds = {r["wav_path"]: float(r["wav_duration_s"]) for r in csv.DictReader(f)}
    table = {}
    with split_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cell = table.setdefault(r["vessel_class"], {}).setdefault(
                r["split"], {"minutes": 0.0, "recordings": 0, "vessels": set()}
            )
            cell["minutes"] += seconds[r["wav_path"]] / 60
            cell["recordings"] += 1
            cell["vessels"].add(r["vessel_group"])
    return {
        cls: {
            split: {
                "minutes": cell["minutes"],
                "recordings": cell["recordings"],
                "vessels": len(cell["vessels"]),
            }
            for split, cell in splits.items()
        }
        for cls, splits in table.items()
    }


def plot_data_minutes(vessels_csv: Path, split_csv: Path, path: Path) -> Path:
    """Grouped bars: minutes of audio per class and split, labelled with vessels / recordings."""
    table = data_minutes(vessels_csv, split_csv)
    classes = [c for c in CLASS_SHORT if c in table]
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    fig.suptitle(
        "Datos: minutos de audio por clase y partición (partición por buque)",
        color=INK,
        fontsize=11,
        x=0.02,
        ha="left",
    )
    width = 0.26
    for s, split in enumerate(SPLIT_ORDER):
        xs = np.arange(len(classes)) + (s - 1) * width
        values = [table[c].get(split, {}).get("minutes", 0.0) for c in classes]
        ax.bar(
            xs,
            values,
            width=width * 0.92,
            color=SPLIT_COLOURS[split],
            label=SPLIT_LABELS[split],
        )
        for x, c, v in zip(xs, classes, values, strict=True):
            cell = table[c].get(split)
            if cell:
                ax.text(
                    x,
                    v + 0.3,
                    f"{v:.1f}\n{cell['vessels']} b · {cell['recordings']} g",
                    ha="center",
                    va="bottom",
                    color=INK,
                    fontsize=6.5,
                )
    ax.set_xticks(range(len(classes)), classes)
    ax.set_ylabel("Minutos de audio")
    ax.set_ylim(
        0, max(max(v["minutes"] for v in t.values()) for t in table.values()) * 1.3
    )
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(
        0.0,
        -0.16,
        "b = buques, g = grabaciones. Ningún buque está en dos particiones.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=7,
    )
    _style(ax)
    return _save(fig, path)


def class_data(vessels_csv: Path) -> dict:
    """Minutes of audio, recordings and vessels of every class, without any split (F5-T11)."""
    import csv

    table = {}
    with vessels_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cell = table.setdefault(
                r["vessel_class"], {"minutes": 0.0, "recordings": 0, "vessels": set()}
            )
            cell["minutes"] += float(r["wav_duration_s"]) / 60
            cell["recordings"] += 1
            cell["vessels"].add(r["vessel_group"])
    return {
        cls: {**cell, "vessels": len(cell["vessels"])} for cls, cell in table.items()
    }


def plot_class_data(vessels_csv: Path, path: Path) -> Path:
    """Bars: minutes of audio per class, labelled with vessels and recordings."""
    table = class_data(vessels_csv)
    classes = [c for c in CLASS_SHORT if c in table]
    fig, ax = plt.subplots(figsize=(7.5, 4), layout="constrained")
    fig.suptitle(
        "Datos: minutos de audio, buques y grabaciones por clase",
        color=INK,
        fontsize=11,
        x=0.02,
        ha="left",
    )
    values = [table[c]["minutes"] for c in classes]
    ax.bar(range(len(classes)), values, width=0.6, color=LINE)
    for x, c, v in zip(range(len(classes)), classes, values, strict=True):
        ax.text(
            x,
            v + 1,
            f"{v:.1f} min\n".replace(".", ",")
            + f"{table[c]['vessels']} buques · {table[c]['recordings']} grabaciones",
            ha="center",
            va="bottom",
            color=INK,
            fontsize=7.5,
        )
    ax.set_xticks(range(len(classes)), classes)
    ax.set_ylabel("Minutos de audio")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    _style(ax)
    return _save(fig, path)


def cv_roles(cv_splits_csv: Path) -> tuple[list[str], list[str], np.ndarray]:
    """(vessels sorted by class, partitions "r.k", roles matrix 0 train / 1 val / 2 test)."""
    import csv

    roles, vessels, partitions = {}, {}, set()
    with cv_splits_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (int(r["repetition"]), int(r["fold"]))
            partitions.add(key)
            vessels[r["vessel_group"]] = r["vessel_class"]
            roles[(r["vessel_group"], key)] = SPLIT_ORDER.index(r["split"])
    order = list(CLASS_SHORT)
    names = sorted(vessels, key=lambda v: (order.index(vessels[v]), v))
    keys = sorted(partitions)
    matrix = np.array([[roles[(v, k)] for k in keys] for v in names])
    return names, [f"{r}.{k}" for r, k in keys], matrix


def plot_cv_partitions(cv_splits_csv: Path, path: Path) -> Path:
    """Matrix vessels x partitions coloured by role: how every vessel rotates in the 5 x 5."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    names, partitions, matrix = cv_roles(cv_splits_csv)
    fig, ax = plt.subplots(figsize=(10, 0.17 * len(names) + 1.8), layout="constrained")
    fig.suptitle(
        "Validación cruzada 5 × 5: papel de cada buque en cada una de las 25 particiones",
        color=INK,
        fontsize=11,
        x=0.02,
        ha="left",
    )
    cmap = ListedColormap([SPLIT_COLOURS[s] for s in SPLIT_ORDER])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(partitions)), partitions, fontsize=6.5, rotation=90)
    ax.set_yticks(range(len(names)), [n.replace("/", " · ") for n in names], fontsize=6)
    ax.set_xlabel("Partición (repetición.vuelta)")
    for x in np.arange(5, len(partitions), 5) - 0.5:  # between repetitions
        ax.axvline(x, color="white", linewidth=2)
    classes = [n.split("/")[0] for n in names]
    for y in [i for i in range(1, len(names)) if classes[i] != classes[i - 1]]:
        ax.axhline(y - 0.5, color="white", linewidth=2)
    ax.legend(
        handles=[
            Patch(color=SPLIT_COLOURS[s], label=SPLIT_LABELS[s]) for s in SPLIT_ORDER
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=False,
        fontsize=8,
    )
    _style(ax)
    return _save(fig, path)
