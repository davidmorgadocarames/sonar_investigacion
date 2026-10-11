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

## 2026-10-11 · F3 Modelo cerrada (F3-T11)
- **Qué se hizo:** segunda pasada del reviewer sobre F3-T12: sin hallazgos críticos ni altos; los
  11 de la primera revisión, resueltos. Con la aprobación de David se arreglaron los 3 medios y 6
  bajos nuevos (commit `5d8d820`): hilos fijados también en `evaluate`, `metrics.json` con
  `"evaluation": 2` y `evaluate --refresh` (actualiza DT-005, permutación e interpretabilidad sin
  puntuar test), comprobación de duración sin azar fijo y regenerada (solo val), línea de azar
  omitida si falta una época, test de `pool_to` con menos valores que grupos. **El test de
  determinismo en CUDA pasó en la imagen GPU** (233 tests en GPU y en CPU). D-020: resúmenes de
  las dos rondas en `docs/results/eval{1,2}.json` (con `tools/summarize_results.py`) y
  `docs/results/README.md` con la comparación y la **tabla de procedencia** (commit y
  `code_sha256` de cada resultado; los de la ronda 2 coinciden con `ab383dd`; la evaluación 2
  corrió con 14 hilos y sin el arreglo B5, sin efecto práctico, no comprobado para no mirar test
  otra vez). DT-005 actualizada con la ronda 2 y las ρ de los tres modelos tratadas igual; nota de
  F5 corregida. Resultado de F3: solo la línea base supera al azar (0,374 frente a 0,238; p =
  0,040), y dentro de cada época ningún modelo se distingue del azar.
- **Qué quedó a medias:** nada de F3. F4 no tiene tareas detalladas.
- **Ficheros tocados:** `tools/summarize_results.py`, `tests/test_summarize_results.py`,
  `docs/results/{eval1.json,eval2.json,README.md}`, `DECISIONS.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm --entrypoint python pipeline -m pytest -rs`
  (235 passed) y leer `docs/results/README.md`.
- **Siguiente paso exacto:** empezar F4 con el procedimiento de inicio de fase: proponer a David
  las tareas (README con opciones A y B, "Comprobar la instalación", subcomando `report`,
  figuras versionadas, limitaciones declaradas, friend-tester, etiqueta `v0.1`, DT-001) y
  esperar su aprobación.

## 2026-10-11 · F3-T12 cerrada: permutación con C dentro y segunda evaluación de test
- **Qué se hizo:** con el visto bueno de David: (5) test de permutación con la elección de C
  dentro de cada permutación, 17,5 min en CPU (se estimaron 8-10): F1 real 0,370 frente a media
  nula 0,272 (p95 0,364; rango 0,161-0,428), 7/200, **p = 0,040** (igual que sin la elección
  dentro); (6) **segunda y última evaluación de test** (30 s): F1 macro por segmento línea base
  0,374 (igual), **CNN 0,115** (antes 0,167), **CNN con aumento 0,252** (antes 0,185); azar
  medido 0,238 (por grabación 0,212; dentro de época 0,493 y 0,468). La CNN lo predice casi todo
  como Passengership; la CNN con aumento, casi todo como Cargo y nunca Tug. Dentro de época:
  CNN con aumento 0,59 en Passengership frente a Tanker y CNN 0,63 en Cargo frente a Tug (IC muy
  anchos). (7) Interpretabilidad (32 s): igual para la línea base; las CNN siguen dependiendo de
  0-40 y 80-120 Hz; solapamiento con las bandas de época 2/5, 2/5 y 1/5. Primera evaluación en
  `outputs/results/eval1/`.
- **Qué quedó a medias:** nada de F3-T12. Falta la nueva pasada del reviewer (F3-T11) y el
  cierre de F3.
- **Ficheros tocados:** `PLAN.md`, `HANDOFF.md` (resultados en `outputs/`, no versionados).
- **Cómo verificarlo:** `docker compose run --rm pipeline evaluate --plot-only` redibuja las
  figuras desde `metrics.json` sin volver a puntuar test.
- **Siguiente paso exacto:** F3-T11: pasar otra vez el reviewer sobre F3-T12 y cerrar F3.

## 2026-10-11 · F3-T12 en curso: arreglos de la revisión y modelos reentrenados
- **Qué se hizo:** revisión de F3 sin hallazgos críticos ni altos; David aprobó arreglar todo lo
  que no daba una salida correcta sin cambiar nada para mejorar la cifra (ajuste 2 de D-018,
  DT-005 reformulada, DT-007 nueva; commit `ab383dd`). Primera evaluación guardada en
  `outputs/results/eval1/` y `outputs/figures/eval1/`. Con el visto bueno de David: línea base
  reentrenada en CPU (pesos idénticos, SHA-256 `1968677a…`); CNN y CNN con aumento reentrenadas
  en GPU (23 s y 15 s). Val: CNN elegida lr=1e-3, wd=1e-2; semillas 0,391 / 0,364 / 0,233 (media
  0,329 ± 0,069); CNN con aumento 0,358 / 0,256 / 0,277 (media 0,297 ± 0,044). El mejor val sigue
  en la época 3 y el lr se reduce a la mitad hacia la época 7.
- **Qué quedó a medias:** por petición de David, parada tras el commit de los modelos: falta el
  test de permutación con C dentro (~8-10 min CPU), la segunda y última evaluación de test y la
  interpretabilidad, con su visto bueno.
- **Ficheros tocados:** `models/{baseline,cnn,cnn_aug}.{npz,pt,json}`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose -f compose.yaml -f compose.gpu.yaml run --rm pipeline
  train --model cnn` (determinista: mismas cifras).
- **Siguiente paso exacto:** con el visto bueno de David, pasos 5-7 de F3-T12.

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
