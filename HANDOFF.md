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

## 2026-10-11 · F2-T07 cerrada: subcomando `features`
- **Qué se hizo:** `src/sonar_pipeline/features.py` y el subcomando `features` (`--data-dir`,
  `--split`, `--output-dir`): lee `split.csv`, preprocesa, calcula LOFAR y DEMON y escribe en
  `datasets/processed/` `features_{train,val,test}.npz` (`segment_id`, `lofar`, `demon`,
  `lofar_hz`, `demon_hz`), `index.csv` y `features_meta.json`. Datos reales: 18 s, 1889 segmentos
  (Cargo 457/129/185, Passengership 225/77/75, Tanker 328/108/109, Tug 67/66/63), 84 MB, sin NaN.
  4 tests de principio a fin con un mini dataset sintético (tono en LOFAR y modulación en DEMON
  por segmento, determinismo, errores). README: sección "Control de calidad y rasgos". David
  aprobó el ajuste de D-015 (`project_commit` null en Docker; `code_sha256`). David pidió que se le
  avise antes de entrenar (nota en F3 de PLAN.md). 130 tests.
- **Qué quedó a medias:** nada. Los `.npz` no son idénticos byte a byte entre ejecuciones (el zip
  guarda la hora); el test compara los arrays.
- **Ficheros tocados:** `src/sonar_pipeline/features.py`, `src/sonar_pipeline/__main__.py`,
  `tests/test_features.py`, `README.md`, `DECISIONS.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm pipeline features` (dice "63 recordings, 1889
  segments").
- **Siguiente paso exacto:** F2-T08: figuras de ejemplo; antes, proponer a David matplotlib como
  dependencia (regla 3, D-002).

## 2026-10-11 · F2-T06 cerrada: DEMON
- **Qué se hizo:** `src/sonar_pipeline/demon.py` (D-015): paso banda Butterworth de orden 4 de 2
  a 8 kHz con `sosfiltfilt`, envolvente |Hilbert| menos su media, diezmado ×8 ×8 con
  `resample_poly` (500 Hz), espectro de magnitud con ventana Hann (301 bins de 1/3 Hz, 0 a
  100 Hz), bins por debajo de 1 Hz a 0 y máximo 1. 23 tests con 3 semillas: un ruido modulado a
  3,4, 7,5, 12 y 50 Hz da el pico en su bin (±1) y más de 20 veces la mediana; con una modulación
  del 10 % se sigue encontrando; sin modulación el pico no pasa de 5 veces la mediana; una
  modulación por debajo de 1 kHz no aparece. Comprobado que, sin el paso banda, el último test
  falla. 127 tests.
- **Qué quedó a medias:** nada. Se usa espectro de magnitud (D-015 no fijaba magnitud o potencia).
- **Ficheros tocados:** `src/sonar_pipeline/demon.py`, `tests/test_demon.py`, `PLAN.md`,
  `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm --entrypoint python pipeline -m pytest -rs
  tests/test_demon.py`.
- **Siguiente paso exacto:** F2-T07: subcomando `features` (lee `split.csv`, escribe los `.npz`,
  `index.csv` y `features_meta.json` en `datasets/processed/`) y README.

## 2026-10-11 · F2-T05 cerrada: LOFAR
- **Qué se hizo:** `src/sonar_pipeline/lofar.py` (D-015): diezmado a 4 kHz con `resample_poly`,
  STFT Hann de 0,5 s / 0,25 s (11 tramas × 1001 bins de 2 Hz), TPSW por trama a lo largo de la
  frecuencia y dB sobre el fondo (suelo numérico de -100 dB). 9 tests: un tono débil (0,1 en ruido
  de 1) a 440, 333 y 1234,5 Hz da el máximo en su bin (±1) en todas las tramas y a más de 10 dB;
  un tono de 3 kHz no se cuela en 1 kHz; el ruido blanco y el rojo quedan planos (±10 % entre
  bandas). Comprobado que los tests fallan con un diezmado sin filtro (alias de 15,6 dB) y sin
  TPSW (ruido rojo 69 veces más fuerte en unas bandas que en otras). 104 tests.
- **Qué quedó a medias:** nada. Nota: el ruido normalizado queda en un nivel constante de unos
  +1,4 dB en potencia media (-0,9 dB de media en dB), no en 0 exacto, por el recorte de la segunda
  pasada del TPSW; no afecta a la forma.
- **Ficheros tocados:** `src/sonar_pipeline/lofar.py`, `tests/test_lofar.py`, `PLAN.md`,
  `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm --entrypoint python pipeline -m pytest -rs
  tests/test_lofar.py`.
- **Siguiente paso exacto:** F2-T06: DEMON con el test de la modulación conocida.

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
