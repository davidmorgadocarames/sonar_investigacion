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
