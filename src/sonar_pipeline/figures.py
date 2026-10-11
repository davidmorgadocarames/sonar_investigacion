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
