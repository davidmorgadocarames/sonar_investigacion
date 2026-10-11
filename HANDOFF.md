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

## 2026-10-11 · F2 Señal cerrada (F2-T09)
- **Qué se hizo:** pasada del reviewer sobre F2: sin hallazgos críticos ni altos; los 2 medios y
  los 9 bajos se arreglaron en F2-T10 (ajuste 2 de D-015 y D-017, aprobados por David). Criterio
  de salida cumplido: un tono conocido aparece en el LOFAR en su bin y una modulación conocida en
  el DEMON en su bin, con controles negativos que hacen fallar los tests si se rompe el
  antialiasing o el paso banda. Entregable: subcomando `features` (1889 segmentos, unos 85 MB en
  `datasets/processed/`) y figuras en `outputs/figures/`.
- **Qué quedó a medias:** nada de F2. F3 no tiene tareas detalladas. Deudas abiertas: DT-001
  (UID, F4) y DT-005 (clase frente a época, a vigilar en F3).
- **Ficheros tocados:** `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm --entrypoint python pipeline -m pytest -rs`
  (152 passed).
- **Siguiente paso exacto:** empezar F3 con el procedimiento de inicio de fase: proponer a David
  las tareas (incluidas las notas de F3 en PLAN.md: pesos por clase, validación cruzada por buque,
  `metrics.json`, DT-005, scikit-learn, PyTorch CPU, dónde se guardan y versionan los pesos) y
  **avisar a David antes de lanzar cualquier entrenamiento**.

## 2026-10-11 · F2-T10 cerrada: hallazgos de la revisión de F2 arreglados
- **Qué se hizo:** el reviewer no dejó hallazgos críticos ni altos; 2 medios y 9 bajos, todos
  arreglados con aprobación de David. M1: LOFAR diezmado a 5 kHz (ajuste 2 de D-015); un tono de
  2100 Hz ya no aparece en 1900 Hz; tests de 2050 a 3000 Hz que fallan con el diezmado antiguo.
  M2: D-017, `requirements.txt` con todas las dependencias (directas y transitivas); un build sin
  caché instala exactamente lo mismo. `features` comprueba el SHA-256 de cada WAV contra
  `vessels.csv`, guarda el SHA de `vessels.csv`, el commit de DeepShip (77725726…) y la versión
  de matplotlib, y borra `.npz` de particiones vacías; figuras solo de train y con el tiempo y la
  frecuencia en su sitio. Tests nuevos: bin exacto, aliasing en DEMON (falla con diezmado sin
  filtro), independencia entre grabaciones, partición del índice. Agentes con `-rs`; README con
  la imagen en unos 630 MB (los 116/147 MB de antes eran tamaños comprimidos). 152 tests.
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `src/sonar_pipeline/{params,lofar,demon,features,figures,__main__}.py`,
  `tests/test_{lofar,demon,features,figures}.py`, `requirements.txt`, `README.md`,
  `.claude/agents/{data-engineer,reviewer}.md`, `DECISIONS.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm --entrypoint python pipeline -m pytest -rs`
  (152 passed) y `docker compose run --rm pipeline features --figures`.
- **Siguiente paso exacto:** cerrar F2 (F2-T09) y empezar F3 con el procedimiento de inicio de
  fase; avisar a David antes de cualquier entrenamiento.

## 2026-10-11 · F2-T08 cerrada: figuras de ejemplo
- **Qué se hizo:** David aprobó D-016 (matplotlib 3.11.2, backend Agg; la imagen pasa de 116 a
  147 MB). `src/sonar_pipeline/figures.py`: LOFARgram con escala secuencial azul (0 a 15 dB sobre
  el fondo) y DEMON medio (desde 1 Hz). `features --figures [--figures-dir]` dibuja la grabación
  más larga de cada clase en `outputs/figures/example_<clase>.png`. Las figuras de validación
  sintética salen de `tests/test_figures.py --figures-dir outputs/figures` (opción en
  `tests/conftest.py`): `validation_lofar.png` (tono de 440 Hz) y `validation_demon.png`
  (modulación de 12 Hz). Revisadas a ojo: Passengership/4 (MALASPINA) muestra en DEMON los
  armónicos del paso de pala cada 11,3 Hz. README actualizado. 136 tests.
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `requirements.txt`, `src/sonar_pipeline/figures.py`,
  `src/sonar_pipeline/features.py`, `src/sonar_pipeline/__main__.py`, `tests/conftest.py`,
  `tests/test_figures.py`, `tests/test_features.py`, `README.md`, `DECISIONS.md`, `PLAN.md`,
  `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm pipeline features --figures` y abrir
  `outputs/figures/`.
- **Siguiente paso exacto:** F2-T09: pasada del reviewer sobre F2 y cierre de fase.
