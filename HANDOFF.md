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

## 2026-10-11 · F3-T04 cerrada: métricas y `metrics.json`
- **Qué se hizo:** `src/sonar_pipeline/metrics.py`: F1 macro y por clase (sobre las 4 clases);
  matriz de confusión en cuentas y normalizada por clase real; ROC una frente al resto (Pfa, Pd,
  AUC, Pd a Pfa = 1 % interpolada); bootstrap de 2000 remuestreos de grabaciones dentro de cada
  clase (percentiles del 95 %); `evaluate_split` por segmento y por grabación; coste (`torch_cost`
  con el contador de FLOPs de PyTorch, `linear_model_cost`); `stopwatch`; hardware (CPU, hilos,
  GPU, versiones); trazabilidad (de `features_meta.json` más un hash del contenido de los arrays,
  porque los bytes del zip cambian); `write_metrics` (JSON estricto, NaN a null, claves
  ordenadas, LF). El esquema de `metrics.json` está en el docstring del módulo. 15 tests con casos
  calculados a mano (uno demuestra que se remuestrean grabaciones y no segmentos). 181 tests.
- **Qué quedó a medias:** nada. Ojo: pre-commit no revisa ficheros nuevos sin `git add -N`.
- **Ficheros tocados:** `src/sonar_pipeline/metrics.py`, `tests/test_metrics.py`, `PLAN.md`,
  `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm --entrypoint python pipeline -m pytest -rs
  tests/test_metrics.py`.
- **Siguiente paso exacto:** F3-T05: clasificador trivial de DT-005 (solo duración de la
  grabación). Es un ajuste de regresión logística de segundos: avisar a David igualmente.

## 2026-10-11 · F3-T03 cerrada: módulo de datos
- **Qué se hizo:** `src/sonar_pipeline/dataset.py`: `CLASSES` en orden fijo (Cargo,
  Passengership, Tanker, Tug); `load_split` carga una partición y falla si un segmento del `.npz`
  es de otra partición en `index.csv`, si falta alguno o si hay clases desconocidas;
  `class_weights` (n / (k · n_k), como `balanced`; falla si falta una clase);
  `baseline_features` (media temporal del LOFAR + DEMON, 1302 columnas); `aggregate_by_recording`
  (media de probabilidades; falla si una grabación mezcla etiquetas). 10 tests, uno sobre los
  rasgos reales: 1077/380/432 segmentos, todas las clases en cada partición, ninguna grabación ni
  buque en dos particiones. 166 tests.
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `src/sonar_pipeline/dataset.py`, `tests/test_dataset.py`, `PLAN.md`,
  `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm --entrypoint python pipeline -m pytest -rs
  tests/test_dataset.py`.
- **Siguiente paso exacto:** F3-T04: módulo de métricas y estructura de `metrics.json`.

## 2026-10-11 · F3-T02 cerrada: scikit-learn y PyTorch (D-019)
- **Qué se hizo:** David aprobó D-019 tras ver su GPU (RTX 4060, 8 GB, driver 610.62).
  scikit-learn 1.9.1 en `requirements.txt` (con cloudpickle, joblib, narwhals, threadpoolctl);
  PyTorch 2.14.1 en `requirements-torch.txt` con sus transitivas, instalado en un paso aparte
  desde `TORCH_INDEX_URL`; `compose.gpu.yaml` pasa de `cu128` (solo llegaba a torch 2.11) a
  `cu130` y documenta la activación con `.env`. Imagen CPU 1,88 GB; imagen GPU 9,44 GB (build de
  7 min), con CUDA disponible y la RTX 4060 visible. `tests/test_dependencies.py` comprueba que lo
  instalado coincide con lo fijado. 156 tests en las dos imágenes.
- **Qué quedó a medias:** nada. La selección automática de dispositivo (`--device`) se implementa
  con el entrenamiento (F3-T07); el README con las opciones A (CPU) y B (GPU con `.env`), en F4.
- **Ficheros tocados:** `requirements.txt`, `requirements-torch.txt`, `Dockerfile`,
  `compose.gpu.yaml`, `tests/test_dependencies.py`, `README.md`, `CLAUDE.md`, `DECISIONS.md`,
  `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose -f compose.yaml -f compose.gpu.yaml run --rm --entrypoint
  python pipeline -c "import torch; print(torch.cuda.is_available())"` (True).
- **Siguiente paso exacto:** F3-T03: módulo de datos (carga de `.npz` e índice, etiquetas, pesos
  por clase, agregación por grabación). Sin entrenamiento.

## 2026-10-11 · F3 empezada; F3-T01 cerrada: diseño del modelo (D-018)
- **Qué se hizo:** David aprobó las 11 tareas de F3 y una F5 opcional (atención, después de F4).
  D-018 recoge el diseño: regresión logística sobre media temporal de LOFAR + DEMON; CNN de dos
  ramas (~80 000 parámetros) con BatchNorm, LeakyReLU, AdamW, lr en coseno, minibatches de 32,
  pesos por clase y parada temprana; CNN con aumento de datos comparada en val; decisión por
  grabación con media de probabilidades; bootstrap por grabaciones; ROC Pd/Pfa; importancia por
  permutación, saliencia y oclusión; pesos en `models/` versionados; `metrics.json` con
  parámetros, FLOPs y tiempos. Descartados SHAP, SwiGLU, RoPE, RMSNorm y GQA.
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `DECISIONS.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** leer D-018 y la sección F3 de PLAN.md.
- **Siguiente paso exacto:** F3-T02: proponer D-019 (scikit-learn y PyTorch CPU). **Antes de
  cualquier entrenamiento o test de permutación, avisar a David con una estimación de tiempo; y
  hacer commit de un modelo entrenado antes de permutar.** Test no se mira hasta F3-T09.
