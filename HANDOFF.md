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

## 2026-10-11 · F4-T05 cerrada: DT-001 pagada (UID y GID del host)
- **Qué se hizo:** `compose.yaml` ejecuta el contenedor con `user: "${HOST_UID:-1000}:${HOST_GID:-1000}"`
  (en Linux, en `.env` con `id -u` e `id -g`; en Windows y macOS no hace falta). La imagen usa
  `HOME=/tmp`, `MPLCONFIGDIR=/tmp/matplotlib` y pytest sin caché, porque un UID del host no tiene
  carpeta personal ni puede escribir en `/app`. Comprobado con el UID 4321: 244 tests en verde y
  `report` escribe en una carpeta nueva ficheros a su nombre (informe idéntico). La prueba también
  reprodujo el problema de DT-001: con un UID distinto no se pueden sobrescribir ficheros creados
  por otro UID. La imagen GPU sigue viendo la RTX 4060. `tests/test_container.py` (2 tests).
  README: paso "Solo en Linux". D-019 con el ajuste del separador (aprobado por David).
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `Dockerfile`, `compose.yaml`, `tests/test_container.py`, `README.md`,
  `DECISIONS.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `HOST_UID=4321 HOST_GID=4321 docker compose run --rm --entrypoint python
  pipeline -m pytest -q`.
- **Siguiente paso exacto:** F4-T06: pasada del friend-tester (clon en la carpeta temporal
  aprobada; opción A; debe reproducir `docs/results/eval2.json`).

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
