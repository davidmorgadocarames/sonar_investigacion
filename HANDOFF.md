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

## 2026-10-11 · F1-T03 cerrada: inventario de DeepShip
- **Qué se hizo:** `data-engineer` clonó la fuente en `datasets/raw/DeepShip` e hizo el
  inventario: son 63 WAV (12/20/28/3), no 67; todos son float32 a 32 kHz y mono; hay 44 buques,
  identificados en los `*-metafile`. David aprobó D-007 (partición por buque), D-008 (numpy y
  scipy), DT-002 (asignación WAV → buque deducida en parte por duración), la corrección del
  recuento en D-005 y `core.autocrlf=false` en D-006.
- **Qué quedó a medias:** nada de F1-T03. El inventario (`outputs/inventory/
  deepship_inventory.csv`) está fuera de git; F1-T06 versionará la tabla WAV → buque.
- **Ficheros tocados:** `DECISIONS.md`, `PLAN.md`, `datasets/README.md`, `HANDOFF.md`.
- **Cómo verificarlo:** D-007, D-008 y DT-002 en [DECISIONS.md](DECISIONS.md); hay 12, 20, 28
  y 3 WAV en las carpetas de `datasets/raw/DeepShip/`.
- **Siguiente paso exacto:** F1-T04: escribir en el README los comandos de descarga y probarlos
  en una carpeta temporal limpia.

## 2026-10-10 · F1-T01 cerrada: fuente y licencia de DeepShip
- **Qué se hizo:** se fijó la fuente (subconjunto público de GitHub, commit `77725726a33d`) y
  se documentó que no tiene licencia explícita (uso académico con cita, sin redistribuir). David
  no tiene el dataset completo: se usa solo el subconjunto.
- **Qué quedó a medias:** nada. Después, el 2026-10-11, David aprobó D-005 (solo el
  subconjunto) y D-006 (descarga con comandos git en el README). F1-T02 está cerrada y F1-T04
  se ha reformulado según D-006.
- **Ficheros tocados:** `datasets/README.md`, `PLAN.md`, `HANDOFF.md`, `DECISIONS.md`.
- **Cómo verificarlo:** leer [datasets/README.md](datasets/README.md) (ya no queda ningún
  "PENDIENTE DE VERIFICAR") y D-005 y D-006 en [DECISIONS.md](DECISIONS.md).
- **Siguiente paso exacto:** lanzar F1-T03 (inventario: ¿los números de fichero identifican
  buques o grabaciones?) con `data-engineer`, sobre un clon de la fuente en
  `datasets/raw/DeepShip`.

## 2026-10-10 · F0 Setup cerrada
- **Qué se hizo:** estructura del repositorio, pre-commit (gitleaks, ruff, check-added-large-files
  con límite de 10 MB, forbid-audio), Docker CPU con usuario no root (UID 1000), override GPU,
  CLI vacía con 5 subcomandos y smoke test, documentos de gobierno y 3 subagentes. Se comprobó
  que un `.wav` y un fichero de 11 MB forzados son rechazados por pre-commit.
- **Qué quedó a medias:** nada de F0. Las propuestas D-002, D-003, D-004 y DT-001 fueron
  aprobadas por David y están apuntadas en DECISIONS.md.
- **Ficheros tocados:** `.gitignore`, `.pre-commit-config.yaml`, `Dockerfile`, `compose.yaml`,
  `compose.gpu.yaml`, `.dockerignore`, `requirements.txt`, `src/sonar_pipeline/*`,
  `tests/test_cli.py`, `CLAUDE.md`, `PLAN.md`, `HANDOFF.md`, `DECISIONS.md`, `README.md`,
  `datasets/README.md`, `.claude/agents/*`, y los `.gitkeep` de `tools/`, `models/`,
  `outputs/` y `datasets/*`.
- **Cómo verificarlo:** los comandos de la sección "Comandos de verificación" de
  [CLAUDE.md](CLAUDE.md).
- **Siguiente paso exacto:** resolver con David las propuestas pendientes de DECISIONS.md y
  esperar a que confirme la licencia de DeepShip (F1-T01). Mientras tanto, se puede lanzar
  F1-T03 con `data-engineer` si David indica dónde está DeepShip en su máquina o servidor.
