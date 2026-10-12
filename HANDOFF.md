# HANDOFF.md

Traspaso entre sesiones. La entrada más reciente va arriba. Se conservan solo las 5 últimas; el
resto queda en el historial de git. El estado de las tareas está en [PLAN.md](PLAN.md); aquí
solo va el contexto para retomar el trabajo.

Plantilla:

```
## AAAA-MM-DD · título
- **Qué se hizo:**
- **Qué quedó a medias:**
- **Ficheros tocados:**
- **Cómo verificarlo:**
- **Siguiente paso exacto:**
```

---

## 2026-10-12 · F5-T06 cerrada: ronda 3 completa (7 modelos, 5 × 5)
- **Qué se hizo:** ronda 3 en la RTX 4060, en tres bloques seguidos (04:51-06:33, unas
  1 h 40 min), sin fallos, con commit y push tras cada bloque (autorizado por David). F1 macro
  de test por segmento (media ± desviación de las 25 particiones; azar medido 0,242):
  línea base (C = 10) 0,331 ± 0,046; CNN (config. C) 0,316 ± 0,135; CNN con aumento (D)
  0,299 ± 0,130; híbrido (B) 0,343 ± 0,126; híbrido con aumento (A) 0,340 ± 0,139; transformer
  (B) 0,235 ± 0,080; transformer con aumento (B) 0,240 ± 0,085. Ninguna de las 13 comparaciones
  es significativa con Holm; la más cercana es transformer frente a línea base (−0,096,
  p = 0,052, Holm 0,68). Dentro de época (azar ≈ 0,5): 2017 (Cargo frente a Tug) híbrido 0,71
  y CNN 0,66; 2016 (Passengership frente a Tanker) todos entre 0,49 y 0,59. Memoria de GPU
  máxima 571 MiB. Todo en `docs/results/round3.json`; pesos representativos en
  `models/round3/representative/`; JSON por entrenamiento en `outputs/round3/` (no versionados).
- **Qué quedó a medias:** nada de F5-T06. Sin interpretar todavía con David.
- **Ficheros tocados:** `docs/results/round3.json`, `models/round3/representative/`, `PLAN.md`,
  `HANDOFF.md`.
- **Cómo verificarlo:** `python -c "import json; print(json.load(open('docs/results/round3.json'))['comparisons'])"`
  y `outputs/round3/run.log`.
- **Siguiente paso exacto:** revisar los resultados con David y empezar F5-T07
  (interpretabilidad de las configuraciones finales sobre validación).

## 2026-10-12 · F5-T03, T04 y T05 cerradas; ronda 3 lanzada (F5-T06)
- **Qué se hizo:** motor de la ronda 3 (`src/sonar_pipeline/cv.py`) y comando único
  `train --round 3` (`--only`, `--measure`); transformer e híbrido
  (`src/sonar_pipeline/transformer.py`: RMSNorm, RoPE en frecuencia sin rotar DEMON ni clase,
  GQA con `expand` determinista, SwiGLU con oculta 8/3 redondeada a múltiplo de 8,
  calentamiento lineal de 5 épocas, mapas de atención); la CNN admite filtros y dropout
  configurables (configuración A idéntica a D-018). JSON por entrenamiento en `outputs/round3/`
  (MLOps, coste con MFLOPs, FLOPs de entrenamiento, segmentos/s, tokens/s, latencia, memoria
  de GPU, curvas, calibración, predicciones por grabación); resumen por modelo y
  `docs/results/round3.json` acumulativo con Nadeau y Bengio + Holm. Pesos según el ajuste 2 de
  D-024 (a + b, aprobado por David): todos en `models/round3/` (ignorados) y uno representativo
  por red versionado. Es reanudable y, si un modelo falla, sigue con los demás. **Detalles de
  implementación que conviene saber:** la semilla 0 del modelo final reutiliza el entrenamiento
  de selección (es el mismo entrenamiento bit a bit; test lo comprueba), así que son 900
  entrenamientos de redes y no 1050; las 13 comparaciones son: cada red frente a la línea base
  (6), con aumento frente a sin aumento (3), y transformer e híbrido frente a CNN, tanto sin
  aumento como con aumento (4). 49 tests nuevos; en la imagen GPU 336 pasan y 1 se salta.
  **Medida previa (partición 0, 2 épocas, RTX 4060):** 0,10 a 0,40 s por época y pico de
  memoria de GPU entre 242 y 568 MiB en las 24 configuraciones: sin riesgo de memoria.
- **Qué quedó a medias:** F5-T06 en marcha por bloques (línea base y CNN; híbrido;
  transformer), con commit y push de `docs/results/round3.json`, los pesos representativos y
  HANDOFF tras cada bloque, autorizado por David.
- **Ficheros tocados:** `src/sonar_pipeline/cv.py`, `src/sonar_pipeline/transformer.py`,
  `src/sonar_pipeline/cnn.py`, `src/sonar_pipeline/__main__.py`, `tests/test_cv.py`,
  `tests/test_transformer.py`, `.gitignore`, `DECISIONS.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose -f compose.yaml -f compose.gpu.yaml run --rm
  --entrypoint python pipeline -m pytest`; `outputs/round3/run.log` durante la ronda.
- **Siguiente paso exacto:** al terminar los tres bloques, revisar `docs/results/round3.json` y
  pasar a F5-T07 (interpretabilidad) con David.

## 2026-10-12 · F5-T03 en curso: plan para hacer T03, T04 y T05 seguidos
- **Qué se hizo:** David pidió hacer F5-T03 (motor de validación cruzada), F5-T04 (transformer)
  y F5-T05 (híbrido) seguidos, y que **todas las preguntas sobre la ronda 3 se le hagan antes**,
  para que después la ronda 3 corra con un solo comando sin acciones suyas (quiere dejarla
  entrenando de noche). Diseño en D-024 y su ajuste 1; particiones en
  `datasets/manifests/cv_splits.csv`.
- **Qué quedó a medias:** F5-T03 recién empezada. **Respuestas de David (2026-10-12):**
  autoriza lanzar la ronda 3 sin esperarle si todos los tests pasan (sin límite de horas dado);
  ejecutar los entrenamientos **en serie**; **medir antes la memoria de GPU** de cada red; si un
  bloque falla, **seguir con los demás** y anotarlo; al terminar, **commit y push automáticos**
  de `docs/results/round3.json` y de la entrada de HANDOFF. Deja el PC encendido con Docker y la
  sesión abiertos. **Pesos:** quiere todos los pesos guardados, cada modelo en su propio
  directorio (p. ej. `models/round3/<modelo>/...`); pendiente de confirmar si se versionan en git.
- **Ficheros tocados:** `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** PLAN.md (F5-T03 en `[~]`).
- **Siguiente paso exacto:** escribir el motor de F5-T03 (`src/sonar_pipeline/cv.py`: entrena
  cualquier modelo en las 25 particiones, elige con media de validación y semilla 0, final con
  semillas 0-2 promediadas, JSON por entrenamiento con MLOps, MFLOPs, tokens/s, tiempos, curvas,
  calibración, predicciones; reanudable; prueba de Nadeau y Bengio con Holm) y sus tests
  sintéticos; luego F5-T04 y F5-T05; un comando único `train --round 3`.

## 2026-10-12 · F5-T02 cerrada: particiones 5 × 5 por buque
- **Qué se hizo:** `data-engineer` creó `tools/make_cv_splits.py` y
  `datasets/manifests/cv_splits.csv` (1575 filas, md5 `3558f6ff…`): 25 particiones (5
  repeticiones × 5 vueltas). Clases con 5 o más buques: 5 folds por repetición con minutos
  equilibrados (tolerancia sobre el mejor reparto), vuelta k con test = fold k y validación =
  fold k + 1; cada buque una vez en test y una en validación por repetición. Tras una primera
  versión en la que Tug faltaba de test en 10 particiones y de validación en otras 10, David
  aprobó que las clases con menos de 5 buques vayan 1/1/resto: Tug 1/1/1 en las 25 (test 9/8/8,
  validación 8/9/8) y la métrica simplificada (F1 macro de 4 clases por partición). Ajuste 1 de
  D-024 con estas reglas, el algoritmo con tolerancia, el desequilibrio de minutos (Cargo 25 %,
  Passengership 42 %, Tanker 10,5 %) y la dependencia del generador de numpy. Sin fugas en
  ninguna partición; 4 clases en los tres papeles en todas; regeneración idéntica. 41 tests
  nuevos (288 en total).
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `tools/make_cv_splits.py`, `datasets/manifests/cv_splits.csv`,
  `tests/test_make_cv_splits.py`, `tests/test_cv_splits_data.py`, `datasets/README.md`,
  `DECISIONS.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm --entrypoint python pipeline
  tools/make_cv_splits.py --summary` y `git status` sin cambios.
- **Siguiente paso exacto:** F5-T03: motor de validación cruzada y JSON por entrenamiento.

## 2026-10-12 · F5 empezada; F5-T01 cerrada: protocolo de la ronda 3 (D-024)
- **Qué se hizo:** David aprobó F5 (9 tareas, sin ARM ni preentrenados) y D-024: validación
  cruzada repetida 5 × 5 por buque (60/20/20, rotando entrenamiento, validación y test; cada buque
  en test 5 veces), 7 modelos (línea base; CNN, transformer puro e híbrido, sin y con aumento),
  4 configuraciones A-D por tipo de red elegidas con la media de validación y 1 semilla, modelo
  final con 3 semillas promediadas, técnicas (AdamW, calentamiento, RMSNorm, RoPE, SwiGLU, GQA),
  prueba de Nadeau y Bengio con Holm, JSON por entrenamiento (MLOps, MFLOPs, tokens/s...), pesos
  de F5 sin versionar, reproducción por el amigo opcional y solo GPU. README: sección "Ronda 3
  (F5, en curso): protocolo de evaluación" con 9 referencias verificadas; "En gráficas" titulada
  como rondas 1 y 2. La reorganización completa por etapas del README va en F5-T08.
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `DECISIONS.md`, `README.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** leer D-024 y la sección del protocolo en el README.
- **Siguiente paso exacto:** F5-T02: particiones 5 × 5 con el subagente data-engineer.
