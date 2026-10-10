# DECISIONS.md

Registro de decisiones y deuda técnica. **Solo se apunta lo que David ha aprobado en el chat**
(regla 2 de [CLAUDE.md](CLAUDE.md)). Los IDs no se reutilizan. Una decisión que cambia otra
anterior se apunta como entrada nueva que la sustituye, y la antigua no se borra.

## Decisiones

### D-001 · Docker con CPU por defecto
- **Fecha:** 2026-10-10
- **Contexto:** un tercero (el amigo de David) debe poder probar el proyecto en su casa con solo
  git y Docker, sin instalar Python ni drivers. El entrenamiento completo se hace en el servidor
  de David, que tiene GPU.
- **Alternativas:** (a) imagen con CUDA por defecto; (b) instalación local con Python y venv;
  (c) imagen CPU por defecto con un override opcional para GPU.
- **Decisión:** (c). `compose.yaml` usa CPU; `compose.gpu.yaml` es un override opcional para
  NVIDIA.
- **Motivo:** la evaluación con pesos ya entrenados no necesita GPU. Una imagen CPU funciona en
  cualquier máquina con Docker y ocupa mucho menos que una con CUDA.

### D-002 · Dependencias añadidas por fase
- **Fecha:** 2026-10-10
- **Contexto:** la regla 3 pide infraestructura mínima y la regla 6, versiones fijadas. En F0 no
  hay código de la pipeline.
- **Alternativas:** (a) fijar ya todo el stack previsto (numpy, scipy, soundfile, torch...);
  (b) añadir cada dependencia en la tarea que la necesita.
- **Decisión:** (b). En F0, `requirements.txt` solo contiene `pytest`.
- **Motivo:** la imagen es pequeña y rápida de construir mientras no hace falta más, y cada
  dependencia queda ligada a la tarea que la justifica.

### D-003 · Carpetas montadas versionadas con `.gitkeep`
- **Fecha:** 2026-10-10
- **Contexto:** `compose.yaml` monta `datasets/`, `models/` y `outputs/`. Si una carpeta montada
  no existe en el host, en Linux Docker la crea como root, y el usuario no root del contenedor
  no puede escribir en ella.
- **Alternativas:** (a) pedir en el README que se creen a mano; (b) versionarlas vacías con un
  `.gitkeep` y que `.gitignore` ignore todo lo demás que contengan.
- **Decisión:** (b), en `datasets/raw/`, `datasets/processed/`, `datasets/manifests/`,
  `models/`, `outputs/` y `tools/`.
- **Motivo:** las carpetas existen nada más clonar, sin pasos manuales, y su contenido sigue
  fuera de git.

### D-004 · Tests dentro de Docker
- **Fecha:** 2026-10-10
- **Contexto:** el amigo solo tiene git y Docker, y el entorno de referencia es la imagen.
- **Alternativas:** (a) tests con un venv local; (b) tests dentro del contenedor.
- **Decisión:** (b): `docker compose run --rm --entrypoint python pipeline -m pytest`. La imagen
  incluye `tests/` y `pytest`.
- **Motivo:** se prueba exactamente el entorno con el que se ejecuta la pipeline, sin depender
  del Python del host.

## Deuda técnica

| ID | Descripción | Por qué se acepta | Impacto | Cuándo pagarla | Estado |
|---|---|---|---|---|---|
| DT-001 | El UID 1000 del usuario `app` está fijo en la imagen. | Es el UID habitual del primer usuario en Linux, y en Docker Desktop (Windows y macOS) no afecta. | En Linux, con otro UID, el contenedor no puede escribir en los volúmenes montados (`outputs/`). | En F4, si el friend-tester lo detecta. | Abierta |
