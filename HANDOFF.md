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

## 2026-10-11 · F4-T04 cerrada: README completo
- **Qué se hizo:** README reescrito: resultado en una frase con enlace al informe; requisitos y
  por qué no hace falta virtualenv (D-021); opción A (CPU) y B (GPU NVIDIA con `.env`);
  "Comprobar la instalación" con `-rs` y qué saltos son normales; datos; QC y rasgos; evaluar
  con los pesos entrenados (cifras que deben salir, idénticas a `docs/results/eval2.json`);
  `report`; reentrenar (tiempos: medidos en GPU y para la línea base, estimados para las CNN en
  CPU; los pesos versionados se entrenaron en GPU); limitaciones; carpetas; cita exacta de
  DeepShip y supuesto de licencia. **Corrección encontrada al probarlo:** en Windows `.env` con
  solo `COMPOSE_FILE=compose.yaml:compose.gpu.yaml` falla (el separador por defecto es `;`);
  hace falta `COMPOSE_PATH_SEPARATOR=:` delante. Comprobado con `docker compose config
  --images` (da `sonar-pipeline:gpu`) y corregido en el README y en `compose.gpu.yaml`. D-019 cita
  la versión antigua: pendiente de que David apruebe el ajuste (regla 2). Sin entrenamientos.
- **Qué quedó a medias:** el ajuste de D-019 y la nota de Linux (UID), que llega con F4-T05.
- **Ficheros tocados:** `README.md`, `compose.gpu.yaml`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** leer el README; `docker compose config --images` con y sin `.env`.
- **Siguiente paso exacto:** F4-T05: pagar DT-001 (UID y GID del host).

## 2026-10-11 · F4-T03 cerrada: subcomando `report` e informe
- **Qué se hizo:** `src/sonar_pipeline/report.py` y `report` (`--results-dir`, `--figures-dir`,
  `--manifests-dir`, `--output`, `--collect-figures DIR`). `report --collect-figures
  outputs/figures` copió una vez 16 figuras a `docs/figures/{signal,eval1,eval2}/` (5,3 MB; las
  de la ronda 1 tal cual); `report` dibuja `docs/figures/data_minutes.png` y escribe
  `docs/report.md` (resumen, datos, señal, modelos y protocolo, las dos rondas con tablas,
  permutación e interpretabilidad, limitaciones y procedencia) solo desde `docs/results/` y los
  manifiestos: no entrena ni puntúa test; regenerarlo da los mismos bytes. `compose.yaml` monta
  `./docs` para que `report` escriba desde Docker. Tabla de procedencia ampliada con las figuras
  de señal (`9f1f99b`, `c867b025…`; el código de la señal no ha cambiado desde entonces) y el
  informe. El test de la CLI ahora exige que todos los subcomandos estén implementados. 243 tests.
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `src/sonar_pipeline/{report,__main__}.py`, `tests/test_report.py`,
  `tests/test_cli.py`, `compose.yaml`, `docs/report.md`, `docs/figures/**`,
  `docs/results/README.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm pipeline report` y comprobar que `git status`
  no cambia.
- **Siguiente paso exacto:** F4-T04: README completo.

## 2026-10-11 · F4-T02 cerrada: figura de minutos por clase y partición
- **Qué se hizo:** `data_minutes` y `plot_data_minutes` en `figures.py`: minutos de audio,
  grabaciones y buques por clase y partición a partir de `vessels.csv` y `split.csv`, sin audio;
  barras agrupadas con los colores 4-6 de la paleta (los 1-3 son los modelos). Dos tests (uno
  sobre los manifiestos reales: 95,4 min, 43 buques, Tug 1/1/1). `report` la guardará en
  `docs/figures/` (F4-T03). 237 tests.
- **Excepción a la regla 9 (aprobada expresamente por David, solo esta vez):** F4-T01 había
  quedado en dos commits publicados (`034ae48` y `28e24a1`, por un fallo del script que escribía
  HANDOFF). Se unieron en uno (`4a44343`, contenido idéntico, mismo árbol) y se publicó con
  `git push --force-with-lease`. La regla 9 sigue vigente.
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `src/sonar_pipeline/figures.py`, `tests/test_figures.py`, `PLAN.md`,
  `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm --entrypoint python pipeline -m pytest -rs
  tests/test_figures.py`.
- **Siguiente paso exacto:** F4-T03: subcomando `report`.

## 2026-10-11 · F4 empezada; F4-T01 cerrada: decisiones de la entrega (D-021)
- **Qué se hizo:** David aprobó las 7 tareas de F4 y D-021: figuras en `docs/figures/` (las de la
  ronda 1 se copian tal cual), informe `docs/report.md` generado por `report` solo desde lo
  versionado, DT-001 pagada ya (UID y GID del host en el compose), etiqueta `v0.1` tras el
  reviewer y **sin virtualenv** (Docker aísla Python y librerías; un virtualenv rompería la
  reproducción exacta). El friend-tester clonará en
  `C:\Users\Gatrix\AppData\Local\Temp\tmp.XXXXXXXXXX\sonar_investigacion` (aprobado; el disco C:
  tiene 79 GB libres).
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `DECISIONS.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** leer D-021 y la sección F4 de PLAN.md.
- **Siguiente paso exacto:** F4-T02: figura de minutos por clase y partición.

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
