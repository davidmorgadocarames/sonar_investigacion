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

## 2026-10-11 · F3-T10 cerrada: interpretabilidad y comprobación de épocas
- **Qué se hizo:** `src/sonar_pipeline/interpret.py` y `evaluate --interpretability` (val, modelos
  congelados, 34 s en CPU): importancia por permutación de bandas (línea base), saliencia y
  oclusión (CNN), comprobación de épocas; resultados en `outputs/results/interpretability.json`,
  mapas en `interpretability_maps.npz`, resumen añadido a `metrics.json` y figura
  `outputs/figures/interpretability_bands.png`. David aprobó anotar en DT-005 la evidencia de
  test. Resultados: **no hay líneas fijas de época** (ninguna en ≥75 % de una época y ≤25 % de la
  otra) ni zumbido de red compartido; las épocas difieren en el nivel medio de algunas bandas
  (1240-1280 Hz 0,99 dB; 80-120 Hz 0,70; 1440-1480 Hz 0,54). La línea base se apoya en 40-80,
  1240-1280 y 1440-1480 Hz: dos de sus tres bandas principales son de las que más separan épocas
  (Spearman +0,29, p 0,039). Las CNN dependen casi solo de 0-40 Hz (y algo de 80-120 Hz), sin
  correlación global con la diferencia entre épocas. Las importancias son pequeñas (≤0,03 de F1;
  ≤0,11 de probabilidad). 214 tests.
- **Qué quedó a medias:** proponer a David anotar esta evidencia en DT-005.
- **Ficheros tocados:** `src/sonar_pipeline/{interpret,figures,__main__}.py`,
  `tests/test_interpret.py`, `PLAN.md`, `HANDOFF.md`, `DECISIONS.md` (nota de test en DT-005).
- **Cómo verificarlo:** `docker compose run --rm pipeline evaluate --interpretability`.
- **Siguiente paso exacto:** F3-T11: pasada del reviewer y cierre de F3.

## 2026-10-11 · F3-T09 cerrada: test evaluado una vez
- **Qué se hizo:** `src/sonar_pipeline/evaluate.py` y `evaluate` (CPU por defecto): carga los
  pesos versionados, puntúa test y escribe `outputs/results/metrics.json` (val, coste, tiempos,
  test por segmento, por grabación y dentro de cada época, DT-005, test de permutación y el
  SHA-256 de cada fichero de pesos: baseline `1968677a…`, cnn `44b4113b…`, cnn_aug
  `20c1bc5a…`) y las figuras `outputs/figures/results_{confusion,f1,roc_segment,roc_recording}.png`.
  Con la aprobación de David, test se evaluó una vez (30 s). F1 macro en test por segmento:
  **línea base 0,374** (IC 0,27-0,47), **CNN 0,167** (0,09-0,23), **CNN con aumento 0,185**
  (0,10-0,28); azar 0,25. La línea base mantiene lo de val (0,381) y la validación cruzada
  (0,370); las CNN caen por debajo del azar: la CNN lo predice casi todo como Passengership y la
  CNN con aumento nunca predice Tug. Dentro de época la línea base queda en el azar
  (Passengership frente a Tanker 0,53; Cargo frente a Tug 0,47): su ventaja viene de separar
  épocas (DT-005). La leyenda de las ROC se movió y las figuras se regeneraron (cifras idénticas,
  evaluación determinista). 209 tests.
- **Qué quedó a medias:** proponer a David cómo anotar en DT-005 la evidencia de test.
- **Ficheros tocados:** `src/sonar_pipeline/{evaluate,figures,__main__}.py`,
  `tests/test_evaluate.py`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm pipeline evaluate` (mismas cifras).
- **Siguiente paso exacto:** F3-T10: interpretabilidad (sobre val) y comprobación de bandas de
  época.

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
