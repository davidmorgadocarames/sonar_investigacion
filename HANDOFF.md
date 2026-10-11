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

## 2026-10-11 · F3-T08 cerrada: CNN con aumento de datos
- **Qué se hizo:** `make_augment` y `train_cnn_aug` en `cnn.py`, `train --model cnn-aug`: misma
  lr y weight decay que `models/cnn.json` y mismas semillas; solo en train y con p = 0,5 cada
  una: 1-2 bandas de LOFAR de hasta 100 Hz, un bloque de 1-2 tramas, desplazamiento circular de
  ±1-2 tramas y una banda de DEMON de hasta 3,3 Hz, rellenas con la media de train.
  Reproducible (comprobado en CUDA). 0,27 s por época en GPU. Con el visto bueno de David se
  entrenó en 14 s. Val: semillas 0,536 / 0,357 / 0,272 (media 0,388 ± 0,110) frente a la CNN sin
  aumento 0,376 / 0,309 / 0,332 (media 0,339 ± 0,028). Modelo guardado (semilla 0): F1 macro
  0,536 por segmento (IC 0,40-0,62) y 0,554 por grabación; dentro de época, Cargo frente a Tug
  0,93 y Passengership frente a Tanker 0,79. **Ojo:** quedarse con la mejor semilla en val
  infla la cifra de val; la comparación justa es la media por semillas, y la mejora (+0,05) es
  menor que la variación entre semillas. Pesos en `models/cnn_aug.{pt,json}`. 207 tests.
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `src/sonar_pipeline/{cnn,__main__}.py`, `tests/test_cnn.py`,
  `tests/test_duration_check.py`, `models/cnn_aug.{pt,json}`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose -f compose.yaml -f compose.gpu.yaml run --rm pipeline
  train --model cnn-aug`.
- **Siguiente paso exacto:** F3-T09: `evaluate`, test una sola vez con los tres modelos (sin
  entrenamiento); `metrics.json` y figuras.

## 2026-10-11 · F3-T07 cerrada: CNN sin aumento entrenada en GPU
- **Qué se hizo:** `src/sonar_pipeline/cnn.py` y `train --model cnn` (`--device`, `--max-epochs`).
  Pooling fijo en vez de adaptativo para que CUDA sea determinista (comprobado en la RTX 4060).
  0,24 s por época en GPU frente a 1,5 s en CPU (datos aleatorios). Con el visto bueno de David
  se entrenó en GPU: 21 s para 4 configuraciones + 2 semillas. Elegida lr=1e-3, wd=1e-3, semilla
  0. Val: F1 macro 0,376 por segmento (IC 0,29-0,43) y 0,365 por grabación, igual que la línea
  base (0,381), pero muy desigual: Tug 0,94, Passengership 0,37, Cargo 0,13, Tanker 0,07; casi
  todo Cargo y Tanker se predice como Passengership. Dentro de época: Cargo frente a Tug 0,83 y
  Passengership frente a Tanker 0,47 por segmento. Semillas: 0,376 / 0,309 / 0,332. **Fuerte
  sobreajuste:** el mejor val llega en la época 3-5 y la pérdida de train baja a 0,06 mientras
  val cae. 79 076 parámetros, 75 MFLOPs por segmento. Pesos en `models/cnn.pt` (328 kB) y
  `models/cnn.json`. 203 tests.
- **Qué quedó a medias:** nada. F3-T08 (aumento de datos) ataca justo el sobreajuste.
- **Ficheros tocados:** `src/sonar_pipeline/{cnn,__main__}.py`, `tests/test_cnn.py`,
  `tests/test_duration_check.py`, `models/cnn.{pt,json}`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose -f compose.yaml -f compose.gpu.yaml run --rm pipeline
  train --model cnn` (mismas cifras: determinista).
- **Siguiente paso exacto:** F3-T08: CNN con aumento; código y tests, estimación de tiempo y
  visto bueno de David antes de entrenar.

## 2026-10-11 · F3-T06 cerrada: test de permutación de la línea base
- **Qué se hizo:** con el visto bueno de David, `train --model baseline --permutation-test 200`
  (3 min 42 s en CPU; scikit-learn no usa GPU). Validación cruzada por buque: F1 macro real 0,370
  frente a 0,270 de media con las clases permutadas entre buques (percentil 95: 0,355; rango
  0,155-0,428); 7 de 200 permutaciones igualan o superan el real: **p = 0,040**. La línea base
  supera al azar, pero por poco; las etiquetas reales incluyen la época (DT-005), así que parte de
  la ventaja puede venir de ahí. Resultado en `outputs/results/permutation_test.json` (no
  versionado; se reproduce con la semilla 0 y entra en `metrics.json` en F3-T09).
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm pipeline train --model baseline
  --permutation-test 200` (mismo p).
- **Siguiente paso exacto:** F3-T07: CNN base sin aumento, en la GPU de David (override
  `compose.gpu.yaml`). Escribir y probar el código con datos sintéticos; medir el tiempo de una
  época sin datos reales; **avisar a David con la estimación antes de entrenar**.

## 2026-10-11 · F3-T06 en curso: línea base entrenada y versionada
- **Qué se hizo:** `src/sonar_pipeline/baseline.py` y `train --model baseline` (opciones
  `--models-dir`, `--bootstrap`, `--permutation-test N`). Con el visto bueno de David se entrenó
  (29 s en CPU): C=10 elegido en val (curva casi plana, 0,34-0,38). Val: F1 macro 0,381 por
  segmento (IC 0,27-0,46) y 0,486 por grabación; dentro de época, Cargo frente a Tug 0,75 por
  segmento y Passengership frente a Tanker 0,58. Validación cruzada por buque en train + val:
  0,370 ± 0,113. 5212 parámetros. Pesos en `models/baseline.npz` (64 kB, sin pickle) y
  `models/baseline.json`, versionados antes del test de permutación, como pidió David.
- **Qué quedó a medias:** el test de permutación (200), pendiente del visto bueno de David con la
  estimación de tiempo (cada validación cruzada tarda ~1 s).
- **Ficheros tocados:** `src/sonar_pipeline/{baseline,__main__}.py`, `tests/test_baseline.py`,
  `tests/test_duration_check.py`, `models/baseline.{npz,json}`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm pipeline train --model baseline` (mismo C y
  mismas cifras).
- **Siguiente paso exacto:** con el visto bueno de David, `train --model baseline
  --permutation-test 200`; después cerrar F3-T06.

## 2026-10-11 · F3-T05 cerrada: comprobación de DT-005 y métrica por época
- **Qué se hizo:** `train --model duration` (`src/sonar_pipeline/duration_check.py`): regresión
  logística solo con la duración de la grabación, ajustada en train y puntuada en val (test no se
  carga); escribe `outputs/results/dt005_duration.json`. Resultado en val: F1 macro 0,93 por
  segmento y 0,80 por grabación (azar ~0,25); la duración separa las épocas ({Cargo, Tug} 2017,
  ~180-200 s; {Passengership, Tanker} 2016, casi todas cortas). David aprobó: DT-005 sigue
  abierta; ajuste de D-018 con métricas dentro de cada época (`ERAS` en `dataset.py`,
  `pair_scores` en `metrics.py`, en `evaluate_split`); comprobación de bandas en F3-T10;
  limitación declarada en F4. Dentro de cada época la duración también acierta en val (Cargo
  frente a Tug 1,00 con solo 3 grabaciones; Passengership frente a Tanker 0,86 por segmento y
  0,60 por grabación): es casualidad de pocos datos y los modelos no ven la duración. `train`
  acepta `baseline`, `cnn` y `cnn-aug`, aún sin implementar. 190 tests.
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `src/sonar_pipeline/{duration_check,dataset,metrics,__main__}.py`,
  `tests/test_{duration_check,metrics,dataset}.py`, `DECISIONS.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm pipeline train --model duration`.
- **Siguiente paso exacto:** F3-T06: línea base (regresión logística). **Avisar a David con una
  estimación de tiempo antes de entrenar**; commit del modelo antes del test de permutación.
