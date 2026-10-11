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

## 2026-10-11 · F2-T04 cerrada: preprocesado
- **Qué se hizo:** `src/sonar_pipeline/params.py` reúne todas las constantes de D-015.
  `src/sonar_pipeline/preprocess.py`: `load_wav` (32 kHz, mono, finito), `remove_dc` (media de
  la grabación entera), `segment` (3 s sin solapamiento, cola descartada, falla si no cabe un
  segmento), `normalize_rms` (RMS = 1, falla con un segmento mudo) y `preprocess`. 15 tests
  nuevos con señales sintéticas y uno sobre los 63 WAV reales: todos dan al menos un segmento y
  salen 1889 (Cargo 771, Passengership 377, Tanker 545, Tug 196), como en D-015. 95 tests.
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `src/sonar_pipeline/params.py`, `src/sonar_pipeline/preprocess.py`,
  `tests/test_preprocess.py`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm --entrypoint python pipeline -m pytest -rs
  tests/test_preprocess.py`.
- **Siguiente paso exacto:** F2-T05: LOFAR (diezmado a 4 kHz, STFT Hann 0,5 s / 0,25 s, TPSW,
  dB sobre el fondo) con los tests del tono y del ruido blanco.

## 2026-10-11 · F2-T03 cerrada: parámetros de la señal (D-015)
- **Qué se hizo:** `.dockerignore` excluye el audio en cualquier carpeta y sin distinguir
  mayúsculas (commit aparte, pedido por David). David aprobó D-015: segmentos de 3 s sin
  solapamiento (1889 en total, Tug 67/66/63), RMS = 1 por segmento, LOFAR a 4 kHz con ventana de
  0,5 s y TPSW (11 × 1001), DEMON de 2 a 8 kHz con envolvente de Hilbert a 500 Hz (301 bins,
  0 a 100 Hz), y un `.npz` por partición más `index.csv` y `features_meta.json`.
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `.dockerignore`, `DECISIONS.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** leer D-015 en DECISIONS.md.
- **Siguiente paso exacto:** F2-T04: módulo de preprocesado (offset DC por grabación, RMS = 1,
  segmentos de 3 s) con tests sintéticos; los parámetros como constantes en un único módulo.

## 2026-10-11 · F2-T02 cerrada: DT-006 pagada
- **Qué se hizo:** `build_vessel_table.py` falla si un id está repetido y si dos bloques empatan
  sin ninguna duración coincidente (data-engineer, D-010). `qc.csv` en LF y orden natural; el test
  de `qc` compara SHA-256; el test de `--help` comprueba exactamente los subcomandos; imagen
  `python:3.11.17-slim`; `requirements.txt` solo desde PyPI (`TORCH_INDEX_URL` queda para PyTorch
  en F3); `.gitignore` sin distinguir mayúsculas; los tests de datos se saltan con `docker run`
  sin compose (70 passed, 10 skipped); `-rs` en CLAUDE.md. DT-006 pagada (aprobado por David).
  80 tests.
- **Qué quedó a medias:** nada. La sección "Comprobar la instalación" del README queda como nota
  en F4 (decisión de David).
- **Ficheros tocados:** `tools/build_vessel_table.py`, `src/sonar_pipeline/qc.py`, `tests/test_qc.py`,
  `tests/test_cli.py`, `tests/test_vessel_table.py`, `tests/test_split_data.py`,
  `tests/test_vessel_table_data.py`, `Dockerfile`, `requirements.txt`, `.gitignore`, `CLAUDE.md`,
  `DECISIONS.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm --entrypoint python pipeline -m pytest -rs`
  (80 passed) y `docker run --rm --entrypoint python sonar-pipeline:latest -m pytest -rs`
  (70 passed, 10 skipped).
- **Siguiente paso exacto:** F2-T03: proponer a David los parámetros de preprocesado, LOFAR,
  DEMON y formato de salida (D-015).

## 2026-10-11 · F2 empezada; F2-T01 cerrada: DT-004 pagada
- **Qué se hizo:** David aprobó las 9 tareas de F2 (en PLAN.md). `data-engineer` creó
  `tests/test_vessel_table_regen.py`: regenera `vessels.csv` desde los 63 WAV en `tmp_path` y lo
  compara byte a byte con el versionado; si falla, muestra el diff y avisa de que regenerar cambia
  grupos y partición; se salta si no hay audios. Comprobado a mano que falla con un grupo
  alterado. DT-004 pagada (aprobado por David). 72 tests.
- **Qué quedó a medias:** nada. David aprobó añadir `-rs` al comando de tests de CLAUDE.md y del
  README dentro de F2-T02 (un verde sin audios no prueba nada sobre los datos).
- **Ficheros tocados:** `tests/test_vessel_table_regen.py`, `DECISIONS.md`, `PLAN.md`,
  `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm --entrypoint python pipeline -m pytest -q -rs`
  (72 passed, 0 skipped con DeepShip descargado).
- **Siguiente paso exacto:** F2-T02: pagar DT-006 (mejoras menores) e incluir `-rs`.

## 2026-10-11 · F1 Datos cerrada (F1-T08)
- **Qué se hizo:** pasada del reviewer sobre F1: sin hallazgos críticos ni altos; 3 medios y 13
  bajos. David aprobó D-014 (`vessels.csv` conserva el extracto de los metafiles bajo el supuesto
  de licencia), DT-004 (test que regenere `vessels.csv`, al empezar F2), DT-005 (clase frente a
  época de grabación, a vigilar en F3) y DT-006 (mejoras menores, en F2). Documentación corregida
  (README: `qc` implementado e imagen de unos 500 MB; PLAN; CLAUDE.md; agente data-engineer;
  "44 nombres de buque"; argumento de que Passengership 37 es COLUMBIA).
- **Qué quedó a medias:** nada de F1. F2 no tiene tareas detalladas.
- **Ficheros tocados:** `DECISIONS.md`, `PLAN.md`, `HANDOFF.md`, `README.md`, `CLAUDE.md`,
  `datasets/README.md`, `.claude/agents/data-engineer.md`.
- **Cómo verificarlo:** los tests en Docker (71) y `docker compose run --rm --entrypoint python
  pipeline tools/check_leaks.py` (OK).
- **Siguiente paso exacto:** empezar F2 con el procedimiento de inicio de fase: proponer a David
  las tareas de F2 (incluidas DT-004 y DT-006, y decidir la longitud de segmento), compararlas con
  PLAN.md y esperar su aprobación. Abrir Claude Code dentro de `sonar_investigacion` para que se
  carguen los subagentes de `.claude/agents/`.
