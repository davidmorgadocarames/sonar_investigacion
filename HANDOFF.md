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

## 2026-10-11 · F1-T06c cerrada: partición por buque
- **Qué se hizo:** `data-engineer` creó `tools/make_split.py` (búsqueda exacta, coste L1 sobre las
  fracciones de minutos por clase) y generó `datasets/manifests/split.csv` (SHA-256
  `51564cf1…b5a7`), aprobado por David: 57/20/23 % de minutos en total, Tug 1/1/1, sin fugas.
  4 tests permanentes sobre el split versionado; 71 tests en total. Notas añadidas en F3 y F4
  (peso de test, empates en Tanker, propuesta de resultados con JSON, ROC Pd/Pfa y SHAP).
- **Qué quedó a medias:** nada. D-013 conserva "con semilla fijada" y lleva una nota de ajuste
  aprobada por David: la búsqueda es exacta y sin azar, así que la semilla no interviene.
- **Ficheros tocados:** `tools/make_split.py`, `datasets/manifests/split.csv`,
  `tests/test_make_split.py`, `tests/test_split_data.py`, `datasets/README.md`, `DECISIONS.md`,
  `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm --entrypoint python pipeline tools/check_leaks.py`
  (OK) y los tests en Docker.
- **Siguiente paso exacto:** F1-T08: pasada del reviewer y cierre de F1.

## 2026-10-11 · F1-T06b cerrada: comprobación de fugas
- **Qué se hizo:** `data-engineer` creó `tools/check_leaks.py` (falla si un buque o una
  grabación, también por SHA-256, aparece en dos particiones, o si el split no cuadra con
  `vessels.csv`) y `--similar-names` para DT-003. Tests de fugas y de DT-002 sobre los datos
  reales (recalcula SHA-256 y duración de los 63 WAV). David aprobó D-011 (BOWEN QUEEN se une a
  "QUEEN OF": 43 grupos) y D-012 (formato de `split.csv`); DT-002 y DT-003 pagadas. 56 tests.
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `tools/check_leaks.py`, `tools/build_vessel_table.py`,
  `tests/test_check_leaks.py`, `tests/test_vessel_table_data.py`, `tests/test_vessel_table.py`,
  `datasets/manifests/vessels.csv`, `datasets/README.md`, `DECISIONS.md`, `PLAN.md`,
  `HANDOFF.md`.
- **Cómo verificarlo:** los tests en Docker y `docker compose run --rm --entrypoint python
  pipeline tools/check_leaks.py --similar-names`.
- **Siguiente paso exacto:** F1-T06c: proponer a David las proporciones de la partición por buque
  (con un buque de Tug en cada partición) antes de generarla.

## 2026-10-11 · F1-T06a cerrada: tabla WAV → buque
- **Qué se hizo:** `data-engineer` creó `tools/build_vessel_table.py`, que genera
  `datasets/manifests/vessels.csv` (63 filas, SHA-256 `ec6ad1a1…79ec`, idéntico al regenerarlo).
  Alineamiento que conserva el orden por clase y bloque: 35 WAV por id, 26 por duración y 2 sin
  verificar (Passengership 30 y 37, grupo COLUMBIA). 44 grupos de buque, ninguno en dos clases.
  `tools/` ahora va dentro de la imagen (Dockerfile y .dockerignore). 17 tests nuevos (33 en
  total).
- **Qué quedó a medias:** tres propuestas pendientes de David (algoritmo de asignación, rutas
  relativas a `raw/DeepShip/`, posibles nombres de buque cortados).
- **Ficheros tocados:** `tools/build_vessel_table.py`, `tests/test_vessel_table.py`,
  `datasets/manifests/vessels.csv`, `datasets/README.md`, `Dockerfile`, `.dockerignore`,
  `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm --entrypoint python pipeline
  tools/build_vessel_table.py` y `sha256sum datasets/manifests/vessels.csv`.
- **Siguiente paso exacto:** F1-T06b: comprobación de fugas y test de DT-002.

## 2026-10-11 · F1-T05 cerrada: subcomando `qc`
- **Qué se hizo:** numpy 2.4.6 y scipy 1.17.1 fijados (D-008). `src/sonar_pipeline/qc.py` mide
  formato, duración, offset DC, saturación, NaN o infinitos y silencio, y escribe
  `outputs/qc/qc.csv` sin excluir nada. 10 tests con señales sintéticas (16 en total, en verde).
  Sobre los 63 WAV: formato correcto, sin recortes, NaN ni silencio; duraciones de 6 a 204 s
  (mediana 47 s); **los 63 tienen un offset DC positivo casi constante (0,027 a 0,032)**, en la
  mayoría mayor que la señal (mediana 6 veces la desviación típica).
- **Qué quedó a medias:** nada. Por acuerdo con David, la comparación de duración con el
  metafile pasó a F1-T06b.
- **Ficheros tocados:** `requirements.txt`, `src/sonar_pipeline/qc.py`,
  `src/sonar_pipeline/__main__.py`, `tests/test_qc.py`, `tests/test_cli.py`, `PLAN.md`,
  `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose build`, los tests y `docker compose run --rm pipeline
  qc` (unos 4 s; dice "63 WAV files checked").
- **Después:** F1-T05b cerrada con D-009 (aprobada): no se excluye ninguna grabación; el offset
  DC se resta en F2. Nota en F3 sobre el desequilibrio de segmentos por clase.
- **Siguiente paso exacto:** F1-T06a: tabla WAV → buque con `data-engineer`.

## 2026-10-11 · F1-T04 cerrada: descarga de DeepShip desde el README
- **Qué se hizo:** se revisaron las tareas pendientes de F1 (T05b, T06a, T06b y T06c nuevas;
  aprobadas por David) y se añadió a CLAUDE.md el procedimiento de inicio y cierre de fase.
  Sección "Datos" del README con dos comandos git y uno de comprobación. El friend-tester los
  siguió en un clon limpio: 58 s de descarga, 63 WAV (12/20/28/3), 68 de 68 ficheros idénticos
  byte a byte y `git status` del proyecto vacío.
- **Qué quedó a medias:** nada de F1-T04. Sugerencias del friend-tester sin aplicar (pendientes
  de David): indicar en el README el tamaño de la imagen (unos 205 MB) y desglosar los 1,2 GB
  (unos 500 MB son `.git`), y medir el build con la caché de Docker vacía (prueba de F4).
- **Ficheros tocados:** `README.md`, `datasets/README.md`, `PLAN.md`, `CLAUDE.md`, `HANDOFF.md`.
  El clon local `datasets/raw/DeepShip` tiene ahora `core.autocrlf=false` (fuera de git).
- **Cómo verificarlo:** seguir la sección "Datos" del [README](README.md) en una carpeta
  temporal; `git -C datasets/raw/DeepShip status` debe decir "working tree clean".
- **Siguiente paso exacto:** F1-T05: añadir numpy y scipy con versión fijada (D-008) e
  implementar `qc` con tests de señales sintéticas.
