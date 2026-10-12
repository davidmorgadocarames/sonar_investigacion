"""Report (F4-T03, D-021; round 3 added in F5-T08): docs/report.md built only from versioned
files.

Inputs: docs/results/eval1.json and eval2.json (summaries of the two F3 rounds, D-020), the
round 3 summaries docs/results/round3.json, round3_details.json and round3_interpretability.json
(F5, when present; their 15 figures are drawn into docs/figures/round3/), the manifests in
datasets/manifests/ and the figures in docs/figures/. Nothing is trained and test is
not scored, so anyone can rebuild the report after cloning, without audio.

`collect_figures` copies once the figures from outputs/figures/ (not versioned) into
docs/figures/: signal/ (examples and synthetic validation), eval1/ (first round, copied as they
are: they cannot be redrawn without scoring test again) and eval2/ (second round).
"""

import json
import shutil
from pathlib import Path

from sonar_pipeline import params
from sonar_pipeline.baseline import C_GRID
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


def position(scores: dict | None) -> str | None:
    """'above' or 'below' when the whole 95 % interval is on one side of the measured chance,
    'chance' when it contains it, None when there is no measured chance (round 1)."""
    if scores is None or scores.get("chance_f1_macro") is None:
        return None
    lo, hi = scores["ci95"]
    if lo > scores["chance_f1_macro"]:
        return "above"
    if hi < scores["chance_f1_macro"]:
        return "below"
    return "chance"


def names(models: list[str]) -> str:
    labels = [MODEL_NAMES.get(m, m) for m in models]
    return (
        labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " y " + labels[-1]
    )


def summary_lines(eval2: dict) -> list[str]:
    """Conclusions read from the numbers: nothing here is written by hand."""
    models = eval2["models"]
    tests = {n: m["test"]["segment"] for n, m in models.items()}
    chance = next(iter(tests.values()))["chance_f1_macro"]
    groups = {
        pos: [n for n, s in tests.items() if position(s) == pos]
        for pos in ("above", "below", "chance")
    }
    lines = []
    if groups["above"]:
        lines.append(
            f"- En test (4 clases, por segmento) supera al azar medido ({num(chance)}) con todo su "
            f"intervalo: "
            + "; ".join(
                f"**{MODEL_NAMES[n]}** {interval(tests[n])}" for n in groups["above"]
            )
            + "."
        )
    else:
        lines.append(
            f"- En test (4 clases, por segmento) ningún modelo supera al azar medido ({num(chance)}) con todo su intervalo."
        )
    rest = [
        f"{MODEL_NAMES[n]} {interval(tests[n])}, "
        + (
            "por debajo del azar"
            if n in groups["below"]
            else "sin distinguirse del azar"
        )
        for n in groups["below"] + groups["chance"]
    ]
    if rest:
        lines.append("- " + "; ".join(rest) + ".")
    permutation = eval2.get("permutation_test")
    cv = models.get("baseline", {}).get("grouped_cv")
    if permutation and cv:
        lines.append(
            f"- Validación cruzada por buque sobre train + val (no es test): línea base "
            f"{num(cv['mean'])} ± {num(cv['std'])}; test de permutación entre buques, "
            f"**p = {num(permutation['p_value'])}**."
        )
    era_above = [
        (n, era)
        for n, m in models.items()
        for era, e in m["test"]["within_era"].items()
        if position(e["segment"]) == "above"
    ]
    if era_above:
        lines.append(
            "- Dentro de una época supera al azar con todo su intervalo: "
            + "; ".join(f"{MODEL_NAMES[n]} en {era}" for n, era in era_above)
            + "."
        )
    else:
        lines.append(
            "- Dentro de cada época de grabación (Cargo frente a Tug, Passengership frente a "
            "Tanker) ningún modelo supera al azar con todo su intervalo: con estos datos **no se "
            "puede afirmar que se reconozca la firma del buque** (ver Limitaciones)."
        )
    return lines


def limitation_lines(eval2: dict) -> list[str]:
    models = eval2["models"]
    lines = []
    duration = eval2.get("dt005_duration_val")
    era_above = any(
        position(e["segment"]) == "above"
        for m in models.values()
        for e in m["test"]["within_era"].values()
    )
    text = (
        "- **Clase y época coinciden** (DT-005): Cargo y Tug se grabaron en 2017 y Passengership "
        "y Tanker en 2016. El resultado de 4 clases no distingue el tipo de buque de las "
        "condiciones de grabación; solo las comparaciones dentro de una época están libres de "
        "ese efecto"
        + (
            ", y ahí ningún modelo supera al azar con todo su intervalo."
            if not era_above
            else "."
        )
    )
    if duration:
        text += (
            f" Un clasificador que solo ve la duración de la grabación obtiene en val un F1 macro "
            f"de {num(duration['segment']['f1_macro'])} por segmento y "
            f"{num(duration['recording']['f1_macro'])} por grabación (azar medido "
            f"{num(duration['segment']['chance_f1_macro'])} y "
            f"{num(duration['recording']['chance_f1_macro'])})."
        )
    lines.append(text)
    lines.append(
        "- **Pocos buques:** 43 en total. Tug tiene 3 (una sola grabación en test): su resultado "
        "no es fiable, y las demás clases también tienen pocos buques."
    )
    unstable = []
    for n, m in models.items():
        rec = m["val"]["recording"]
        if position(rec) == "below" or (
            rec.get("chance_f1_macro") is not None
            and rec["f1_macro"] < rec["chance_f1_macro"]
        ):
            unstable.append(
                f"la {MODEL_NAMES[n]} guardada da en val, por grabación, "
                f"{num(rec['f1_macro'])} frente a un azar de {num(rec['chance_f1_macro'])}"
            )
    seeds = [
        f"{MODEL_NAMES[n]} {num(m['val_f1_seed_mean'])} ± {num(m['val_f1_seed_std'])}"
        for n, m in models.items()
        if "val_f1_seed_mean" in m
    ]
    if unstable or seeds:
        lines.append(
            "- **Modelos inestables:** "
            + "; ".join(
                unstable
                + ([f"F1 de val entre semillas: {', '.join(seeds)}"] if seeds else [])
            )
            + "."
        )
    lines += [
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
    ]
    c = models.get("baseline", {}).get("config", {}).get("c")
    lines.append(
        "- La validación cruzada de la línea base mezcla folds de 3 y 4 clases (Tug falta en 3 de "
        "5)"
        + (
            f", y el C elegido ({c:g}) está en el borde de la rejilla."
            if c in (min(C_GRID), max(C_GRID))
            else "."
        )
    )
    lines.append(
        "- **Reproducción exacta comprobada en x86-64** (Intel Core i7-14700KF, imagen CPU); en "
        "otros procesadores, sobre todo ARM, puede variar la última cifra."
    )
    return lines


def build_report(
    eval1: dict, eval2: dict, data_lines: list[str], round3: tuple | None = None
) -> str:
    """round3: (round3.json, round3_details.json, round3_interpretability.json) or None."""
    from sonar_pipeline import report_round3

    final = eval2["models"]
    if round3 is not None:
        summary3, details3, interpretability3 = round3
        top = [
            "### Ronda 3 (F5): evaluación principal",
            "",
            *report_round3.summary_lines(summary3),
            "",
            "### Rondas 1 y 2 (F3): partición fija, histórico",
            "",
        ]
        round3_lines = report_round3.round3_section(
            summary3, details3, interpretability3, "round3"
        )
        round3_models = [
            (
                "- **Ronda 3 (F5):** además, un **transformer** (bandas de 40 Hz del LOFAR, un "
                "token del DEMON y uno de clase; RMSNorm, RoPE, GQA y SwiGLU) y un **híbrido** "
                "(dos bloques de la CNN como tokens y bloques transformer), cada red sin y con "
                "aumento, con 4 configuraciones cada una y validación cruzada 5 × 5 por buque "
                "(D-024). Arquitecturas y justificación de cada configuración en el README."
            ),
        ]
        numbers = ("5", "6", "7")
        historic_title = "## 5. Rondas 1 y 2 (F3): partición fija · histórico"
        historic_note = (
            "**Etapa anterior, no comparable directamente con la ronda 3:** una sola partición "
            "fija (12 buques en test), mirada dos veces. "
        )
    else:
        top, round3_lines, round3_models = [], [], []
        numbers = ("4", "5", "6")
        historic_title = "## 4. Resultados"
        historic_note = ""
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
        *top,
        *summary_lines(eval2),
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
        *round3_models,
        "",
        *round3_lines,
        historic_title,
        "",
        (
            historic_note
            + "F1 macro con intervalos del 95 % (bootstrap remuestreando grabaciones dentro de cada "
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
            "se suponía 0,25 y 0,5, y la Pd a Pfa = 1 % de sus ROC está interpolada con menos de "
            "100 negativos).",
        ),
        f"## {numbers[1]}. Limitaciones",
        "",
        *limitation_lines(eval2),
        *(report_round3.limitation_lines(round3[0]) if round3 is not None else []),
        "",
        f"## {numbers[2]}. Procedencia y reproducibilidad",
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
    round3 = None
    if (results_dir / "round3.json").is_file():
        from sonar_pipeline.figures_round3 import plot_round3

        round3 = tuple(
            json.loads((results_dir / f"{name}.json").read_text(encoding="utf-8"))
            for name in ("round3", "round3_details", "round3_interpretability")
        )
        plot_round3(*round3, figures_dir / "round3")
    text = build_report(eval1, eval2, data_section(manifests_dir, figures_dir), round3)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")
    return output
