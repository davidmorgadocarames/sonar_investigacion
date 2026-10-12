"""Round 3 chapter of the report (F5-T08): built from docs/results/round3*.json only.

Every number and every comparison word ("supera", "no se distingue") is read from the
summaries; the explanations of how to read each figure are fixed text.
"""

import numpy as np

from sonar_pipeline.figures_round3 import FIGURES, NAMES, ORDER, corrected_interval


def num(value, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}".replace(".", ",")


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def era_label(name: str) -> str:
    if name.endswith("_attention"):
        return NAMES[name.removesuffix("_attention")] + " (atención)"
    return NAMES[name]


def mean_std(entry: dict, digits: int = 3) -> str:
    return f"{num(entry['mean'], digits)} ± {num(entry['std'], digits)}"


def models_of(summary: dict) -> list[str]:
    return [m for m in ORDER if m in summary["models"]]


def results_table(summary: dict, details: dict) -> list[str]:
    lines = [
        (
            "| Modelo | Config. | Val (elección) | **Test, 4 clases** | Mín.-máx. | Test por grabación "
            "| 2016: Pasaje vs Tanker | 2017: Cargo vs Tug | ECE | MFLOPs | Parámetros |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in models_of(summary):
        e = summary["models"][m]
        t = e["test"]
        f1 = t["f1_macro_segment"]
        eras = t["within_era_segment"]
        val = e["selection"][e["chosen_config"]]["val_f1_macro_mean"]
        lines.append(
            f"| {NAMES[m]} | {e['chosen_config']} | {num(val)} | **{mean_std(f1)}** | "
            f"{num(f1['min'], 2)}-{num(f1['max'], 2)} | {mean_std(t['f1_macro_recording'])} | "
            f"{mean_std(eras['2016'])} | {mean_std(eras['2017'])} | "
            f"{num(details['models'][m]['calibration']['ece'], 2)} | "
            f"{num(e['cost']['mflops_per_segment'], 2)} | {thousands(e['cost']['parameters'])} |"
        )
    chance = np.mean(
        [
            summary["models"][m]["test"]["chance_f1_macro"]["mean"]
            for m in models_of(summary)
        ]
    )
    lines.append(
        f"| *Azar medido* | | | *{num(chance)}* | | | *≈ 0,5* | *≈ 0,5* | | | |"
    )
    return lines


def comparisons_table(summary: dict) -> list[str]:
    lines = [
        "| Comparación | Diferencia media | IC 95 % corregido | Gana en | p | p (Holm) |",
        "|---|---|---|---|---|---|",
    ]
    for c in summary["comparisons"]:
        lo, hi = corrected_interval(c)
        lines.append(
            f"| {NAMES[c['model']]} − {NAMES[c['against']]} | {num(c['mean_difference'])} | "
            f"{num(lo)} a {num(hi)} | {c['wins']}/{c['partitions']} | {num(c['p'])} | "
            f"{num(c['p_holm'])} |"
        )
    return lines


def summary_lines(summary: dict) -> list[str]:
    """Conclusions of round 3 read from the numbers."""
    models = models_of(summary)
    f1 = {m: summary["models"][m]["test"]["f1_macro_segment"] for m in models}
    best = max(models, key=lambda m: f1[m]["mean"])
    worst = min(models, key=lambda m: f1[m]["mean"])
    chance = np.mean(
        [summary["models"][m]["test"]["chance_f1_macro"]["mean"] for m in models]
    )
    significant = [c for c in summary["comparisons"] if c["p_holm"] < 0.05]
    lines = [
        (
            f"- **Ronda 3 (evaluación principal, validación cruzada 5 × 5 por buque):** F1 macro de "
            f"test por segmento entre {num(f1[worst]['mean'])} ({NAMES[worst]}) y "
            f"{num(f1[best]['mean'])} ({NAMES[best]}), con un azar medido de {num(chance)}."
        ),
    ]
    if significant:
        lines.append(
            "- Diferencias significativas tras la corrección de Holm: "
            + "; ".join(
                f"{NAMES[c['model']]} − {NAMES[c['against']]} {num(c['mean_difference'])} "
                f"(p Holm {num(c['p_holm'])})"
                for c in significant
            )
            + "."
        )
    else:
        closest = min(summary["comparisons"], key=lambda c: c["p"])
        lines.append(
            f"- **Ninguna de las {len(summary['comparisons'])} comparaciones fijadas de antemano "
            f"es significativa** con la prueba de Nadeau y Bengio y la corrección de Holm. La "
            f"más cercana: {NAMES[closest['model']]} − {NAMES[closest['against']]}, "
            f"{num(closest['mean_difference'])} (p {num(closest['p'])}, Holm "
            f"{num(closest['p_holm'])})."
        )
    nets = [m for m in models if m != "baseline"]
    if "baseline" in models and nets:
        spread = {m: f1[m]["std"] for m in models}
        lines.append(
            f"- Las redes varían más entre particiones (desviación de "
            f"{num(min(spread[m] for m in nets), 2)} a {num(max(spread[m] for m in nets), 2)}) "
            f"que la línea base ({num(spread['baseline'], 2)}): su resultado depende de qué "
            "buques caen en test."
        )
    return lines


FIGURE_TEXT = {
    "box": (
        "F1 de test de cada partición",
        (
            "Cada punto es una de las 25 particiones; la caja va del primer al tercer cuartil, la "
            "raya es la mediana y el rombo la media. La línea discontinua es el azar (medido con 4 "
            "clases; 0,5 aproximado dentro de cada época, donde solo hay 2 clases). Lo importante "
            "es lo **alta** que es cada caja: dice cuánto cambia el resultado según qué buques le "
            "tocan al modelo en test."
        ),
    ),
    "pairs": (
        "Comparaciones fijadas de antemano",
        (
            "Diferencia media de F1 entre dos modelos en las mismas 25 particiones, con su intervalo "
            "del 95 % corregido por Nadeau y Bengio (que tiene en cuenta que las particiones "
            "comparten buques). Si el intervalo cruza el 0, la diferencia no se distingue del "
            "ruido. A la derecha, el p sin corregir y con Holm (13 comparaciones a la vez)."
        ),
    ),
    "augmentation": (
        "Efecto del aumento de datos",
        (
            "Para cada partición, F1 con aumento menos F1 sin aumento de la misma arquitectura. Por "
            "encima de 0, el aumento ayudó en esa partición."
        ),
    ),
    "cost": (
        "Precisión frente a coste",
        (
            "F1 medio (± desviación) frente al coste de inferencia, al tamaño y al tiempo de "
            "entrenamiento, en escala logarítmica. Un modelo compensa si está más arriba sin irse "
            "mucho a la derecha."
        ),
    ),
    "speed": (
        "Velocidad y memoria",
        (
            "Medidas en la RTX 4060 (la línea base, en CPU). Los tokens por segundo solo existen en "
            "el transformer y el híbrido. 'Formar minibatches' es la parte del tiempo de "
            "entrenamiento que se va en preparar cada lote: casi toda es el aumento de datos, así "
            "que solo pesa en las versiones con aumento."
        ),
    ),
    "curves": (
        "Curvas de aprendizaje",
        (
            "Media (y banda de ± 1 desviación) de los entrenamientos finales, época a época. Si la "
            "pérdida de entrenamiento baja mientras la de validación sube, el modelo está "
            "memorizando: es el sobreajuste. El último panel es la norma del gradiente (media por "
            "época, en escala logarítmica): si crece, los pasos del entrenamiento se hacen más "
            "grandes y menos estables."
        ),
    ),
    "generalisation": (
        "Brecha de generalización",
        (
            "F1 del modelo final (media de sus semillas) en entrenamiento, validación y test. La "
            "distancia entre el círculo y el rombo es lo que el modelo memoriza y no generaliza."
        ),
    ),
    "variance": (
        "De dónde viene la variación",
        (
            "Desviación del F1 de test entre particiones (con la media de semillas) y entre "
            "semillas dentro de una misma partición. Es el análisis de Bouthillier et al. (2021): "
            "justifica usar muchas particiones antes que muchas semillas."
        ),
    ),
    "selection": (
        "Elección de configuración",
        (
            "F1 de validación (media ± desviación de las 25 particiones) de cada configuración; la "
            "rellena es la elegida y el rombo su F1 de test. Si las barras se solapan mucho, la "
            "elección es casi un empate."
        ),
    ),
    "confusion": (
        "Matrices de confusión conjuntas",
        (
            "Todas las predicciones de test de las 25 particiones juntas (cada segmento cuenta 5 "
            "veces, una por repetición). Filas: clase real; columnas: clase predicha."
        ),
    ),
    "vessels": (
        "Acierto por buque",
        (
            "Fracción de las veces que cada buque estuvo en test en que sus grabaciones se "
            "clasificaron bien. Muestra los buques que todos los modelos fallan."
        ),
    ),
    "agreement": (
        "Acuerdo entre modelos",
        (
            "Para cada pareja de modelos, en qué fracción de las grabaciones de test predicen lo "
            "mismo y en qué fracción fallan los dos. Si dos modelos fallan en sitios distintos, "
            "combinarlos podría ayudar."
        ),
    ),
    "calibration": (
        "Calibración",
        (
            'Si el modelo dice "80 % seguro", ¿acierta el 80 % de las veces? La diagonal es la '
            "calibración perfecta; por debajo, el modelo está demasiado seguro. ECE y Brier: "
            "cuanto más bajos, mejor."
        ),
    ),
    "interpretability": (
        "Qué frecuencias usa cada modelo",
        (
            "Arriba, cuánto depende cada modelo de cada banda de 40 Hz (normalizado por fila), y "
            "en la última fila cuánto difiere esa banda entre las grabaciones de 2016 y 2017. Si un "
            "modelo usa las bandas que delatan la época, puede estar aprendiendo la época y no el "
            "buque. Abajo, a qué bandas atiende el token de clase del transformer y del híbrido."
        ),
    ),
    "roc": (
        "ROC de clasificación",
        (
            "Una clase frente al resto, con los segmentos de test de las 25 particiones juntos. Es "
            "una ROC de clasificación, no de detección (DT-007). Al juntar particiones se mezclan "
            "25 modelos distintos, así que el AUC es orientativo."
        ),
    ),
}


def interpretability_lines(report: dict) -> list[str]:
    lines = []
    for m in [m for m in ORDER if m in report["models"]]:
        entry = report["models"][m]
        top = ", ".join(
            f"{b['lo_hz']:g}-{b['hi_hz']:g}" for b in entry["top_lofar_bands"]
        )
        extra = ""
        if entry.get("attention_demon_mean") is not None:
            share = entry["attention_demon_mean"][-1]
            extra = f"; atención del token de clase al DEMON {num(100 * share, 0)} %"
        lines.append(f"- {NAMES[m]}: bandas principales {top} Hz{extra}.")
    era = report["era_check"]
    lines.append(
        "- Bandas que más difieren entre épocas: "
        + ", ".join(f"{h:g}" for h in era["top_difference_bands_hz"])
        + " Hz; líneas fijas de época: "
        + (", ".join(f"{h:g} Hz" for h in era["era_lines_hz"]) or "ninguna")
        + "."
    )
    lines.append(
        "- Coincidencia de las 5 bandas principales con esas 5: "
        + "; ".join(
            f"{era_label(n)} {m['top_overlap']}/5" for n, m in era["models"].items()
        )
        + ". Las correlaciones con la diferencia entre épocas no son fiables (bandas vecinas "
        "correlacionadas, muchos modelos sin corrección)."
    )
    return lines


def selection_table(summary: dict) -> list[str]:
    lines = []
    for m in models_of(summary):
        e = summary["models"][m]
        cells = ", ".join(
            f"{'**' if c == e['chosen_config'] else ''}{c} {num(s['val_f1_macro_mean'])}"
            f"{'**' if c == e['chosen_config'] else ''}"
            for c, s in e["selection"].items()
        )
        lines.append(f"- {NAMES[m]}: {cells}.")
    return lines


def retrain_lines(initial: dict, summary: dict) -> list[str]:
    """Notice of the F5-T10 retraining and the initial results next to the retrained ones."""
    lines = [
        (
            "**Resultados reentrenados (F5-T10).** Los resultados iniciales de la ronda 3 tenían un "
            "fallo del calentamiento del lr en el transformer y el híbrido, sin y con aumento: al "
            "terminar las 5 épocas de calentamiento el lr se quedaba en el 82 % (4,1e-4 en lugar de "
            "5e-4; ajuste 3 de D-024). Se arregló y se reentrenó la ronda completa **después de "
            "haber visto test**: igual que la ronda 2, es un arreglo de implementación decidido "
            "por la revisión del código, no por las cifras. La CNN y la línea base no tenían "
            "calentamiento. Los resultados iniciales se conservan en "
            "`docs/results/round3_initial/`."
        ),
        "",
        (
            "| Modelo | Config. inicial | Config. reentrenada | Test inicial | Test reentrenado | "
            "Diferencia |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for m in models_of(summary):
        if m not in initial["models"]:
            continue
        old, new = initial["models"][m], summary["models"][m]
        f_old = old["test"]["f1_macro_segment"]
        f_new = new["test"]["f1_macro_segment"]
        lines.append(
            f"| {NAMES[m]} | {old['chosen_config']} | {new['chosen_config']} | "
            f"{mean_std(f_old)} | {mean_std(f_new)} | {num(f_new['mean'] - f_old['mean'])} |"
        )
    return lines


def round3_section(
    summary: dict,
    details: dict,
    interpretability: dict,
    folder: str,
    initial: dict | None = None,
) -> list[str]:
    lines = [
        "## 4. Ronda 3 (F5): validación cruzada 5 × 5 · evaluación principal",
        "",
        *(retrain_lines(initial, summary) + [""] if initial is not None else []),
        (
            f"Protocolo fijado antes de entrenar (D-024 y sus ajustes): {summary['partitions']} "
            "particiones 60/20/20 por buque (5 repeticiones × 5 folds; Tug, con 3 buques, 1/1/1 en "
            "todas); la configuración de cada modelo se elige con la media de validación de las "
            "25 particiones y la semilla 0, sin mirar test; el modelo final promedia las "
            f"probabilidades de las semillas {', '.join(map(str, summary['seeds']))}. Ningún "
            "modelo está preentrenado. Comparaciones con la prueba t corregida de Nadeau y Bengio "
            "(relación test/entrenamiento "
            f"{num(summary['test_train_ratio'], 3)}) y corrección de Holm."
        ),
        "",
        "F1 macro de test por segmento (media ± desviación de las 25 particiones):",
        "",
        *results_table(summary, details),
        "",
        "Comparaciones fijadas de antemano:",
        "",
        *comparisons_table(summary),
        "",
        "Configuraciones (F1 medio de validación; en negrita, la elegida):",
        "",
        *selection_table(summary),
        "",
        "Interpretabilidad (sobre validación, modelos finales):",
        "",
        *interpretability_lines(interpretability),
        "",
    ]
    for key, (title, text) in FIGURE_TEXT.items():
        lines += [
            f"### {title}",
            "",
            f"![{title}](figures/{folder}/{FIGURES[key]})",
            "",
            text,
            "",
        ]
    return lines


def limitation_lines(summary: dict) -> list[str]:
    return [
        (
            "- **Ronda 3, Tug:** con 3 buques, cada uno cae en test 8 o 9 veces y siempre con los "
            "otros dos en entrenamiento y validación; el solapamiento entre particiones es mayor "
            "en esa clase que el que corrige la prueba de Nadeau y Bengio (ajuste 1 de D-024)."
        ),
        (
            "- **Ronda 3, azar dentro de época:** se marca en 0,5 aproximado; el azar medido de "
            "cada pareja no se guardó por partición."
        ),
        (
            "- **Ronda 3, figuras conjuntas** (matrices, ROC, calibración): juntan predicciones de "
            "25 modelos distintos; las cifras por partición son las de referencia."
        ),
    ]
