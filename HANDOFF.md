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

## 2026-10-12 · F5-T09: revisión pasada; fallo del lr declarado; F5-T10 obligatoria
- **Qué se hizo:** reviewer y friend-tester sobre `a6c9aac`. El friend-tester siguió el README en
  limpio en CPU sin atascos (cifras y reproducción idénticas; `report` sin cambios). El reviewer
  no encontró fugas, secretos ni audio, pero sí un **fallo alto**: el calentamiento del lr del
  transformer y del híbrido (sin y con aumento) se quedaba en el 82 % (4,1e-4 en lugar de 5e-4)
  tras la época 5 (`cv.py`, bucle de `fit_network`); la CNN y la línea base no se ven afectadas.
  Con David: se declara ya (ajuste 3 de D-024, aprobado, que recoge además las 13 comparaciones,
  la semilla 0 reutilizada y la semántica de ReduceLROnPlateau) y un aviso de resultados
  provisionales en el README; **F5-T10 pasa a obligatoria**: arreglar el lr, guardar la norma del
  gradiente y el tiempo de carga de datos, y reentrenar; los resultados reentrenados serán los de
  la ronda 3 y los actuales, histórico; el reentrenamiento se declara como la ronda 2. **`v0.2`
  se etiqueta después de F5-T10.** README ampliado: aviso, por qué hay calentamiento solo en
  transformer e híbrido (Liu et al. 2020, Vaswani et al. 2017, Ioffe y Szegedy 2015), qué
  modelos intervienen en cada sección, tabla de tests saltados, tiempo CPU frente a GPU, y las
  propuestas del friend-tester (rutas largas en Windows, qué regenera `report`, MFLOPs, ECE) y
  los arreglos de documentación del reviewer. **David pide aviso antes de cualquier uso de la
  GPU** para no usar el PC (memoria actualizada).
- **Qué quedó a medias:** 3 cambios de código ya revisados con David, sin commit, para F5-T10:
  ayuda de `--measure` (`__main__.py`), dispositivo, hardware y etiqueta `split` en
  `interpret_cv.py` y dispositivo y hardware en `round3_details.py`. El aviso de reentrenamiento
  en el informe va en F5-T10 (el informe se genera desde el código).
- **Ficheros tocados:** `README.md`, `DECISIONS.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** leer el ajuste 3 de D-024 y el aviso al principio del README.
- **Siguiente paso exacto:** F5-T10 según el plan aprobado (B1-B13): código y tests sin GPU;
  después avisar a David con el tiempo medido antes de usar la GPU.

## 2026-10-12 · F5-T08 cerrada: gráficas, informe y README de la ronda 3
- **Qué se hizo:** `evaluate --round 3 --details` (`round3_details.py`, 1 min en GPU, sin
  entrenar) reúne en `docs/results/round3_details.json` lo que las gráficas necesitaban y solo
  estaba en `outputs/`: velocidad, curvas, F1 de entrenamiento y validación por inferencia,
  variación por partición y por semilla, acierto por buque, acuerdo entre modelos, calibración
  (ECE, Brier, log loss) y ROC. `figures_round3.py` dibuja las 15 gráficas acordadas en
  `docs/figures/round3/`, y `report_round3.py` añade al informe el capítulo de la ronda 3
  (cifras leídas de los JSON) y las rondas 1 y 2 como histórico. README reorganizado por etapas:
  resumen con el resultado de la ronda 3, cronología, índice, uso común, cómo reproducir la ronda
  3, datos y señal, ronda 3 (protocolo con literatura, **arquitecturas explicadas**: CNN,
  transformer e híbrido con BatchNorm, LeakyReLU, RMSNorm, RoPE, GQA, SwiGLU, AdamW, *weight
  decay*, calentamiento, decaimiento del lr, parada temprana y aumento; **justificación de cada
  configuración** con la hipótesis que prueba y su F1 de validación; las 5 gráficas principales
  explicadas, con "4. Resultados en test" como diagrama de caja), rondas 1 y 2 como histórico
  no comparable, limitaciones y procedencia. `evaluate` admite `--outputs-dir`. Nuevo dato: la
  línea base acierta el 100 % de entrenamiento y es la peor calibrada (ECE 0,45). No se
  guardaron la norma del gradiente ni los tiempos de carga de datos que se mencionaron al
  planificar: no se pueden recuperar sin reentrenar.
- **Qué quedó a medias:** nada de F5-T08.
- **Ficheros tocados:** `src/sonar_pipeline/{round3_details,figures_round3,report_round3,report,__main__}.py`,
  `tests/test_report_round3.py`, `docs/results/round3_details.json`, `docs/results/README.md`,
  `docs/figures/round3/`, `docs/report.md`, `README.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm --entrypoint python pipeline -m pytest -rs`
  (347 pasan y 4 se saltan en CPU; 350 y 1 en GPU); `docker compose run --rm pipeline report`
  no cambia `docs/report.md`.
- **Siguiente paso exacto:** F5-T09: reviewer, friend-tester con el README nuevo, cierre de F5
  y etiqueta `v0.2`.

## 2026-10-12 · F5-T07 cerrada: interpretabilidad de la ronda 3
- **Qué se hizo:** `src/sonar_pipeline/interpret_cv.py` y `evaluate --round 3
  --interpretability --results-dir docs/results` (9,5 min en la RTX 4060, solo inferencia, solo
  validación de cada partición). Por modelo y partición, con la configuración elegida y sus
  semillas finales (media de semillas dentro de la partición; media y desviación sobre las 25):
  línea base, importancia por permutación; redes, oclusión por banda y saliencia por banda y
  clase; transformer e híbrido, atención del token de clase a DEMON y a cada banda (posiciones
  del híbrido llevadas a bandas de 40 Hz por su frecuencia central). Comprobación de época de
  F3-T10 con todos los buques. Resultado en `docs/results/round3_interpretability.json`.
  **Lo principal:** todas las redes dependen sobre todo de 0-40 Hz (y el híbrido y el
  transformer también de 40-80 Hz), donde dominan el ruido del mar y de la plataforma; la línea
  base, de 1240-1320 Hz, y 3 de sus 5 bandas principales están entre las 5 que más difieren entre
  épocas (las redes, 1 de 5). No hay líneas fijas de época. La atención del token de clase al
  DEMON va del 11 % (transformer) al 19 % (híbrido con aumento). Correlaciones con la diferencia
  entre épocas pequeñas (ρ de −0,12 a +0,39) y sus p no son fiables (bandas vecinas
  correlacionadas, muchos modelos sin corrección).
- **Qué quedó a medias:** nada de F5-T07. Interpretación con David en F5-T08.
- **Ficheros tocados:** `src/sonar_pipeline/interpret_cv.py`, `src/sonar_pipeline/__main__.py`,
  `tests/test_interpret_cv.py`, `docs/results/round3_interpretability.json`, `PLAN.md`,
  `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose -f compose.yaml -f compose.gpu.yaml run --rm
  --entrypoint python pipeline -m pytest -rs` (343 pasan, 1 saltado).
- **Siguiente paso exacto:** F5-T08: gráficas, informe y README por etapas, con la sección de
  arquitecturas y la justificación de cada configuración.

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
