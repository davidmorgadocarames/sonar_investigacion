"""Report (F4-T03, D-021): docs/report.md built only from versioned files.

Inputs: docs/results/eval1.json and eval2.json (summaries of the two F3 rounds, D-020), the
manifests in datasets/manifests/ and the figures in docs/figures/. Nothing is trained and test is
not scored, so anyone can rebuild the report after cloning, without audio.

`collect_figures` copies once the figures from outputs/figures/ (not versioned) into
docs/figures/: signal/ (examples and synthetic validation), eval1/ (first round, copied as they
are: they cannot be redrawn without scoring test again) and eval2/ (second round).
"""

import json
import shutil
from pathlib import Path

from sonar_pipeline import params
from sonar_pipeline.dataset import CLASSES

FIGURE_GROUPS = {
    "signal": [
        "example_cargo.png",
        "example_passengership.png",
        "example_tanker.png",
        "example_tug.png",
        "validation_lofar.png",
        "validation_demon.png",
    ],
    "eval2": [
        "results_f1.png",
        "results_confusion.png",
        "results_roc_segment.png",
        "results_roc_recording.png",
        "interpretability_bands.png",
    ],
    "eval1": [
        "eval1/results_f1.png",
        "eval1/results_confusion.png",
        "eval1/results_roc_segment.png",
        "eval1/results_roc_recording.png",
        "eval1/interpretability_bands.png",
    ],
}
MODEL_NAMES = {
    "baseline": "Línea base",
    "cnn": "CNN sin aumento",
    "cnn_aug": "CNN con aumento",
}
SPLIT_NAMES = {"train": "Entrenamiento", "val": "Validación", "test": "Test"}


def num(value: float | None, digits: int = 3) -> str:
    """Spanish decimal comma; a dash for missing values."""
    return "—" if value is None else f"{value:.{digits}f}".replace(".", ",")


def interval(scores: dict | None) -> str:
    if scores is None:
        return "—"
    lo, hi = scores["ci95"]
    return f"{num(scores['f1_macro'])} ({num(lo, 2)}-{num(hi, 2)})"


def collect_figures(source: Path, figures_dir: Path) -> list[Path]:
    """Copy the figures of outputs/figures into docs/figures/{signal,eval1,eval2}."""
    copied = []
    for group, names in FIGURE_GROUPS.items():
        for name in names:
            origin = source / name
            if not origin.is_file():
                raise FileNotFoundError(f"{origin} not found")
            target = figures_dir / group / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)
            copied.append(target)
    return copied


def data_section(manifests_dir: Path, figures_dir: Path) -> list[str]:
    from sonar_pipeline.figures import data_minutes, plot_data_minutes

    vessels, split = manifests_dir / "vessels.csv", manifests_dir / "split.csv"
    plot_data_minutes(vessels, split, figures_dir / "data_minutes.png")
    table = data_minutes(vessels, split)
    total = sum(c["minutes"] for s in table.values() for c in s.values())
    n_vessels = sum(c["vessels"] for s in table.values() for c in s.values())
    n_recordings = sum(c["recordings"] for s in table.values() for c in s.values())
    lines = [
        "## 1. Datos",
        "",
        (
            f"Subconjunto público de DeepShip ({n_recordings} grabaciones, {n_vessels} buques, "
            f"{num(total, 1)} minutos de audio; 4 clases). La partición es **por buque**: ningún "
            "buque aparece en dos particiones (D-007, D-013)."
        ),
        "",
        "| Clase | " + " | ".join(SPLIT_NAMES[s] for s in SPLIT_NAMES) + " |",
        "|---|" + "---|" * len(SPLIT_NAMES),
    ]
    for cls in CLASSES:
        cells = []
        for split in SPLIT_NAMES:
            c = table.get(cls, {}).get(split)
            cells.append(
                f"{num(c['minutes'], 1)} min · {c['vessels']} b · {c['recordings']} g"
                if c
                else "—"
            )
        lines.append(f"| {cls} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "b = buques, g = grabaciones.",
        "",
        "![Minutos por clase y partición](figures/data_minutes.png)",
        "",
    ]
    return lines


def signal_section() -> list[str]:
    return [
        "## 2. Señal",
        "",
        (
            f"Cada grabación se centra (offset DC) y se corta en segmentos de {num(params.SEGMENT_S, 0)} s "
            "sin solapamiento, escalados a RMS = 1 (D-015)."
        ),
        "",
        (
            f"- **LOFAR:** diezmado a {params.LOFAR_SAMPLE_RATE / 1000:g} kHz, ventana Hann de "
            f"{num(params.LOFAR_WINDOW_S, 2)} s con avance de {num(params.LOFAR_HOP_S, 2)} s, "
            f"0-{params.LOFAR_MAX_HZ:g} Hz (11 tramas × 1001 bins de 2 Hz), normalizado con TPSW "
            "(dB sobre el fondo)."
        ),
        (
            f"- **DEMON:** paso banda {params.DEMON_BAND_HZ[0]:g}-{params.DEMON_BAND_HZ[1]:g} Hz, "
            f"envolvente de Hilbert diezmada a 500 Hz, espectro de 0 a {params.DEMON_MAX_HZ:g} Hz "
            "(301 bins de 1/3 Hz)."
        ),
        "",
        (
            "Validación con señales sintéticas: un tono de 440 Hz aparece en su bin del LOFAR y una "
            "modulación de 12 Hz en el del DEMON."
        ),
        "",
        "![Validación LOFAR](figures/signal/validation_lofar.png)",
        "![Validación DEMON](figures/signal/validation_demon.png)",
        "",
        "Ejemplos (la grabación de entrenamiento más larga de cada clase):",
        "",
        *[f"![Ejemplo {c}](figures/signal/example_{c.lower()}.png)" for c in CLASSES],
        "",
    ]


def results_table(summary: dict) -> list[str]:
    lines = [
        (
            "| Modelo | Val (segmento) | Val, media de semillas | Test (segmento) | "
            "Test (grabación) | Test 2016: Passeng. vs Tanker | Test 2017: Cargo vs Tug |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in summary["models"].items():
        seeds = (
            f"{num(m['val_f1_seed_mean'])} ± {num(m['val_f1_seed_std'])}"
            if "val_f1_seed_mean" in m
            else "—"
        )
        eras = m["test"]["within_era"]
        lines.append(
            f"| {MODEL_NAMES.get(name, name)} | {num(m['val']['segment']['f1_macro'])} | "
            f"{seeds} | {interval(m['test']['segment'])} | {interval(m['test']['recording'])} | "
            f"{interval(eras['2016']['segment'])} | {interval(eras['2017']['segment'])} |"
        )
    reference = next(iter(summary["models"].values()))["test"]
    chance = reference["segment"].get("chance_f1_macro")
    if chance is not None:
        eras = reference["within_era"]
        lines.append(
            f"| *Azar medido* | | | *{num(chance)}* | "
            f"*{num(reference['recording']['chance_f1_macro'])}* | "
            f"*{num(eras['2016']['segment']['chance_f1_macro'])}* | "
            f"*{num(eras['2017']['segment']['chance_f1_macro'])}* |"
        )
    else:
        lines.append("| *Azar (supuesto)* | | | *0,25* | *0,25* | *0,5* | *0,5* |")
    return lines


def permutation_line(summary: dict) -> str:
    p = summary.get("permutation_test")
    if not p:
        return ""
    # Round 2 chooses C inside every permutation (D-018 adjustment 2); round 1 kept it fixed.
    procedure = (
        "eligiendo C con val dentro de cada permutación"
        if p.get("procedure")
        else "C fijo"
    )
    return (
        f"Test de permutación de la línea base ({p['n_permutations']} permutaciones de las "
        f"clases entre buques; {procedure}): F1 real {num(p['observed_f1_macro'])} frente a "
        f"una media nula de {num(p['null_mean'])} (percentil 95: {num(p['null_p95'])}); "
        f"**p = {num(p['p_value'])}**."
    )


def interpretability_lines(summary: dict) -> list[str]:
    i = summary.get("interpretability")
    if not i:
        return []
    lines = [
        f"- Línea base, bandas de LOFAR más importantes: {', '.join(i['baseline_top_lofar_bands'])}.",
    ]
    for name, bands in i["cnn_top_lofar_occlusion"].items():
        lines.append(f"- {MODEL_NAMES.get(name, name)}, oclusión: {', '.join(bands)}.")
    era_lines = ", ".join(f"{h:g} Hz" for h in i["era_lines_hz"]) or "ninguna"
    lines.append(f"- Líneas fijas de época: {era_lines}.")
    lines.append(
        "- Relación con las bandas que más difieren entre épocas: "
        + "; ".join(
            f"{MODEL_NAMES.get(n, n)} ρ = {num(m['spearman_rho'], 2)} "
            f"(p = {num(m['spearman_p'], 3)}), {m['top_overlap']}/5 bandas en común"
            for n, m in i["era_check_by_model"].items()
        )
        + ". Ninguno de estos p es fiable (bandas vecinas correlacionadas, sin corrección).",
    )
    return lines


def round_section(summary: dict, title: str, folder: str, note: str) -> list[str]:
    return [
        f"### {title}",
        "",
        note,
        "",
        *results_table(summary),
        "",
        permutation_line(summary),
        "",
        "Interpretabilidad (sobre val):",
        "",
        *interpretability_lines(summary),
        "",
        f"![F1 con intervalos](figures/{folder}/results_f1.png)",
        f"![Matrices de confusión](figures/{folder}/results_confusion.png)",
        f"![ROC por segmento](figures/{folder}/results_roc_segment.png)",
        f"![ROC por grabación](figures/{folder}/results_roc_recording.png)",
        f"![Interpretabilidad por bandas](figures/{folder}/interpretability_bands.png)",
        "",
    ]


def build_report(eval1: dict, eval2: dict, data_lines: list[str]) -> str:
    final = eval2["models"]
    base = final["baseline"]["test"]["segment"]
    chance = base.get("chance_f1_macro")
    permutation = eval2.get("permutation_test")
    p_text = (
        f" (p = {num(permutation['p_value'])} en el test de permutación)"
        if permutation
        else ""
    )
    lines = [
        "# Informe de resultados · sonar_investigacion",
        "",
        (
            "*Generado por `python -m sonar_pipeline report` a partir de `docs/results/` "
            "(D-020, D-021). No editar a mano.*"
        ),
        "",
        "## Resumen",
        "",
        (
            f"- Solo la **línea base** (regresión logística sobre LOFAR y DEMON promediados) supera "
            f"al azar en test: F1 macro {num(base['f1_macro'])} frente a un azar medido de "
            f"{num(chance)}{p_text}."
        ),
        (
            f"- Las dos CNN no generalizan: {num(final['cnn']['test']['segment']['f1_macro'])} sin "
            f"aumento y {num(final['cnn_aug']['test']['segment']['f1_macro'])} con aumento."
        ),
        (
            "- Dentro de cada época de grabación (Cargo frente a Tug, Passengership frente a Tanker) "
            "ningún modelo se distingue del azar: con estos datos **no se puede afirmar que se "
            "reconozca la firma del buque** (ver Limitaciones)."
        ),
        "",
        *data_lines,
        *signal_section(),
        "## 3. Modelos y protocolo",
        "",
        (
            "- **Línea base:** regresión logística estandarizada con pesos por clase; C elegido con "
            "val; validación cruzada agrupada por buque y test de permutación entre buques."
        ),
        (
            "- **CNN:** dos ramas (LOFARgram 2D y DEMON 1D), unos 79 000 parámetros, AdamW, pesos por "
            "clase y parada temprana con el F1 macro de val; rejilla 2 × 2 y 3 semillas."
        ),
        (
            "- **CNN con aumento:** la misma red, configuración y semillas, con enmascarado de "
            "frecuencia y tiempo y pequeños desplazamientos solo en entrenamiento."
        ),
        (
            "- Todo se decide con val; test se usa solo para la evaluación final de cada ronda. La "
            "decisión por grabación es la media de las probabilidades de sus segmentos (D-018)."
        ),
        "",
        "## 4. Resultados",
        "",
        (
            "F1 macro con intervalos del 95 % (bootstrap remuestreando grabaciones dentro de cada "
            "clase). Hubo **dos rondas**: la segunda tras arreglar los fallos de implementación y de "
            "medida que encontró la revisión de F3, sin cambiar nada para mejorar la cifra; test se "
            "había mirado ya en la primera. Detalle y procedencia en "
            "[docs/results/README.md](results/README.md)."
        ),
        "",
        *round_section(
            eval2,
            "Ronda 2 (final)",
            "eval2",
            "Modelos versionados en `models/` (los que usa `evaluate`).",
        ),
        *round_section(
            eval1,
            "Ronda 1",
            "eval1",
            "Primera evaluación de test, antes de los arreglos (figuras copiadas tal cual; el azar "
            "se suponía 0,25 y 0,5).",
        ),
        "## 5. Limitaciones",
        "",
        (
            "- **Clase y época coinciden** (DT-005): Cargo y Tug se grabaron en 2017 y Passengership "
            "y Tanker en 2016. El resultado de 4 clases no distingue el tipo de buque de las "
            "condiciones de grabación; solo las comparaciones dentro de una época están libres de "
            "ese efecto, y ahí ningún modelo supera al azar."
        ),
        (
            "- **Pocos buques:** 43 en total. Tug tiene 3 (una sola grabación en test): su resultado "
            "no es fiable, y las demás clases también tienen pocos buques."
        ),
        (
            "- **La ROC es de clasificación** (una clase frente al resto), no de detección: DeepShip "
            "no tiene grabaciones de solo fondo (DT-007). La Pd a Pfa = 1 % no se puede medir con "
            "menos de 100 negativos."
        ),
        (
            "- **Azar medido:** con clases desequilibradas el azar no es 0,25 exacto; se mide con un "
            "clasificador aleatorio sobre las etiquetas reales."
        ),
        (
            "- **Dos evaluaciones de test:** la segunda se hizo después de ver la primera; los "
            "arreglos vinieron de la revisión del código, no de las cifras."
        ),
        (
            "- La validación cruzada de la línea base mezcla folds de 3 y 4 clases (Tug falta en 3 de "
            "5), y el C elegido (10) está en el borde de la rejilla."
        ),
        "",
        "## 6. Procedencia y reproducibilidad",
        "",
        (
            "- Qué código produjo cada resultado: tabla de procedencia en "
            "[docs/results/README.md](results/README.md)."
        ),
        "- SHA-256 de los pesos evaluados (ronda 2): "
        + ", ".join(f"{n} `{m['weights_sha256'][:12]}…`" for n, m in final.items())
        + ".",
        (
            "- Las figuras de señal se generaron con `features --figures` y los tests de figuras; las "
            "de resultados, con `evaluate` (la ronda 2 redibujada con `evaluate --plot-only`)."
        ),
        "",
    ]
    return "\n".join(line for line in lines) + "\n"


def run_report(
    results_dir: Path,
    figures_dir: Path,
    manifests_dir: Path,
    output: Path,
    collect_from: Path | None = None,
) -> Path:
    if collect_from is not None:
        collect_figures(collect_from, figures_dir)
    eval1 = json.loads((results_dir / "eval1.json").read_text(encoding="utf-8"))
    eval2 = json.loads((results_dir / "eval2.json").read_text(encoding="utf-8"))
    text = build_report(eval1, eval2, data_section(manifests_dir, figures_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")
    return output
