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

## Deuda técnica

| ID | Descripción | Por qué se acepta | Impacto | Cuándo pagarla | Estado |
|---|---|---|---|---|---|
| — | (sin entradas) | | | | |
