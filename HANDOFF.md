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

## 2026-10-10 · F0 Setup cerrada
- **Qué se hizo:** estructura del repositorio, pre-commit (gitleaks, ruff, check-added-large-files
  con límite de 10 MB, forbid-audio), Docker CPU con usuario no root (UID 1000), override GPU,
  CLI vacía con 5 subcomandos y smoke test, documentos de gobierno y 3 subagentes. Se comprobó
  que un `.wav` y un fichero de 11 MB forzados son rechazados por pre-commit.
- **Qué quedó a medias:** nada de F0. Hay 3 decisiones y 1 deuda propuestas a David en el chat
  y pendientes de aprobación, todavía no apuntadas en DECISIONS.md: dependencias por fase,
  `.gitkeep` en las carpetas montadas, tests en Docker, y el UID 1000 fijo.
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
