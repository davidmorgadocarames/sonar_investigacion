# CLAUDE.md · Orquestador de sonar_investigacion

Proyecto pequeño de toma de contacto con el procesado de señal sonar pasivo (DeepShip, 4 clases:
Cargo, Passengership, Tanker, Tug). Pipeline objetivo:

```
WAV → control de calidad → preprocesado (offset DC, normalización, segmentación)
    → LOFAR y DEMON → clasificador → evaluación → informe con figuras
```

Este fichero se lee en cada sesión. Cada dato vive en un solo documento:

| Documento | Qué contiene |
|---|---|
| [PLAN.md](PLAN.md) | Fases, tareas y su estado. **Única fuente de estado.** |
| [HANDOFF.md](HANDOFF.md) | Traspaso entre sesiones (5 últimas entradas). |
| [DECISIONS.md](DECISIONS.md) | Decisiones y deuda técnica aprobadas por David. |
| [README.md](README.md) | Instrucciones para un tercero que solo tiene git y Docker. |
| [datasets/README.md](datasets/README.md) | Fuentes, licencias y obtención de datos. |

## Rol

La sesión principal es el **orquestador**: coordina, escribe el código y verifica. Los
subagentes existen para trabajar con un contexto limpio; no pueden lanzar otros subagentes.

## Reglas que no se negocian

1. **Coordinación.** La lleva el orquestador, que también escribe el código. Los subagentes no
   delegan en otros subagentes.
2. **DECISIONS.md.** Antes de escribir una decisión o una deuda técnica, se propone en el chat y
   se espera la aprobación de David. Si no la aprueba, no se apunta.
3. **Infraestructura mínima.** No se añade ninguna herramienta que no pida una tarea concreta.
4. **Audio.** Ningún audio entra en git sin aprobación expresa. Lo impiden `.gitignore` y el
   hook `forbid-audio` de pre-commit. Cómo le llegan los datos al amigo está en D-006.
5. **Particiones.** train/val/test se hacen **por buque** si los metadatos lo permiten; si no,
   por grabación. **Nunca por segmento.** El manifiesto con la partición se versiona en
   `datasets/manifests/`.
6. **Docker.** CPU por defecto; la GPU es un override opcional (`compose.gpu.yaml`). El
   contenedor corre como usuario no root. Las versiones se fijan en `requirements.txt` y
   `requirements-torch.txt`; PyTorch se instala aparte desde https://download.pytorch.org/whl/cpu
   (o `cu130` con el override de GPU, D-019).
7. **Sin GitHub Actions** por ahora; se discutirá antes. Seguridad shift-left con pre-commit:
   gitleaks, ruff y check-added-large-files.
8. **Idiomas.** Documentación en español. Nombres de ficheros, código y commits en inglés, con
   Conventional Commits. Un commit por tarea terminada.
9. **Push.** Push a `origin/main` al terminar cada tarea verificada, con pre-commit en verde.
   Nunca `--force`, nunca `--no-verify` y nunca reescribir historial publicado. Si un push
   falla, se para y se explica a David por qué.

## Subagentes (`.claude/agents/`)

| Subagente | Cuándo usarlo | Puede escribir |
|---|---|---|
| `data-engineer` | Obtención de datos, control de calidad, manifiestos, particiones y comprobación de fugas. | Sí |
| `reviewer` | Al cerrar una fase o antes de una entrega: fugas, secretos, audio en git, licencias, coherencia entre documentos y código, y resultados que salgan de verdad de `outputs/`. | No |
| `friend-tester` | Cuando el README cambia de forma relevante y en F4: clona en limpio y sigue el README al pie de la letra. | No |

Todos devuelven: resumen · ficheros tocados · cómo verificarlo · decisiones o deudas propuestas.
Las propuestas pasan por la regla 2 antes de llegar a DECISIONS.md.

## Criterio de "hecho"

Una tarea está hecha cuando se cumplen las cuatro cosas:
- Los tests pasan.
- `docker compose build` termina sin errores.
- PLAN.md está actualizado.
- El commit está hecho y publicado en `origin/main`.

## Ciclo de sesión

1. Leer [HANDOFF.md](HANDOFF.md), empezando por la entrada de arriba.
2. Leer [PLAN.md](PLAN.md) y elegir la siguiente tarea pendiente de la fase actual.
3. Marcarla `[~]` y hacerla, o delegarla en el subagente que corresponda.
4. Verificar con los comandos de abajo.
5. Proponer en el chat las decisiones o deudas que hayan surgido (regla 2).
6. Actualizar PLAN.md (`[x]`, o `[!]` con el motivo).
7. Escribir la entrada de HANDOFF.md.
8. Hacer el commit y el push, y comprobar que `git log origin/main -1` coincide con `HEAD`.

## Inicio y cierre de fase

- **Al empezar una fase:** leer CLAUDE.md, HANDOFF.md y PLAN.md. Proponer a David las tareas de
  la fase, comparándolas con las que haya en PLAN.md, y esperar su aprobación antes de empezar.
- **Durante la fase:** tarea a tarea: hacerla o delegarla, verificar su criterio de aceptación,
  commit y push.
- **Al cerrar la fase:** pasar el `reviewer` y enseñar sus hallazgos a David; proponer las
  decisiones o deudas surgidas (regla 2); actualizar PLAN.md y HANDOFF.md.
- **Lo que no esté en ninguna tarea se pregunta a David antes de hacerlo.**

## Comandos de verificación

```bash
docker compose build
docker compose run --rm pipeline --help
docker compose run --rm --entrypoint python pipeline -m pytest -rs  # -rs: lista los saltados
python -m pre_commit run --all-files     # en esta máquina, `pre-commit` no está en el PATH
git status --short --untracked-files=all  # no debe aparecer ningún audio
```

## Convenciones de código

- Paquete en `src/sonar_pipeline/`. Se ejecuta con `python -m sonar_pipeline <command>`, y en
  la imagen `PYTHONPATH=/app/src`.
- Scripts de apoyo (descarga, checksums, fugas) en `tools/`.
- Las señales sintéticas solo existen en `tests/`. **No hay modo demo con datos sintéticos.**
- Volúmenes: `datasets/`, `models/` y `outputs/`. Los pesos se versionan en `models/` solo si
  ocupan menos de 10 MB.
