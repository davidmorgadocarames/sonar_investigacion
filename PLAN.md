# PLAN.md

**Única fuente de estado del proyecto.** Solo se detalla en tareas la fase actual; las demás
quedan a alto nivel.

- **Formato de tarea:** `- [ ] F1-T01 · descripción · responsable · criterio de aceptación verificable`
- **Estados:** `[ ]` pendiente · `[~]` en curso · `[x]` hecho · `[!]` bloqueado (con motivo)
- **Fase actual:** **F1 Datos**

| Fase | Nombre | Estado |
|---|---|---|
| F0 | Setup | [x] |
| F1 | Datos | [ ] |
| F2 | Señal | [ ] |
| F3 | Modelo | [ ] |
| F4 | Entrega | [ ] |

---

## F0 · Setup — [x] hecha (2026-10-10)

- **Objetivo:** estructura, documentos de gobierno, subagentes y un Docker mínimo, sin código de
  la pipeline.
- **Entregable:** repositorio con CLI vacía que se construye y se ejecuta en Docker.
- **Criterio de salida:** `docker compose build`, `docker compose run --rm pipeline --help`, los
  tests y `pre-commit run --all-files` en verde; ningún audio en git; último commit en
  `origin/main`.

- [x] F0-T01 · Ampliar `.gitignore` y pre-commit (ruff, check-added-large-files, forbid-audio) · orquestador · un `.wav` y un fichero de más de 10 MB forzados con `git add -f` son rechazados
- [x] F0-T02 · Dockerfile, compose, override GPU y CLI vacía con smoke test · orquestador · build OK, `--help` lista los 5 subcomandos, 6 tests pasan, el contenedor corre con UID 1000
- [x] F0-T03 · CLAUDE.md, DECISIONS.md, README.md, datasets/README.md y subagentes · orquestador · ficheros en commit; CLAUDE.md con menos de 120 líneas
- [x] F0-T04 · PLAN.md y HANDOFF.md, con F1 detallada · orquestador · último commit en `origin/main`

## F1 · Datos — [ ] fase actual

- **Objetivo:** tener DeepShip localizado, verificado, con control de calidad y particionado sin
  fugas, y decidir cómo le llegan los datos al amigo.
- **Entregable:** script de obtención con checksums, subcomando `qc`, manifiesto de partición
  versionado y comprobación de fugas.
- **Criterio de salida:** licencia confirmada; decisión de distribución aprobada; manifiesto en
  `datasets/manifests/`; la comprobación de fugas pasa; la pasada del reviewer no deja
  hallazgos graves abiertos.

- [x] F1-T01 · Confirmar la licencia y la fuente oficial de DeepShip · David · licencia y URL anotadas en `datasets/README.md`, sin "PENDIENTE DE VERIFICAR"
- [x] F1-T02 · Proponer cómo le llegan los datos al amigo (depende de F1-T01) · orquestador → aprobación de David · decisión aprobada y apuntada en DECISIONS.md (D-005 y D-006)
- [ ] F1-T03 · Inventariar DeepShip: estructura de carpetas, metadatos y si identifican el buque · data-engineer · informe con conteos por clase, buque y grabación, y unidad de partición elegida (buque o grabación) con su justificación
- [ ] F1-T04 · Comandos de descarga en el README (git clone + checkout del commit fijado, D-006) · orquestador · probados en una carpeta temporal limpia: dejan 67 WAV en `datasets/raw/DeepShip` (13/21/29/4 por clase) y `git status` del proyecto no muestra ningún audio
- [ ] F1-T05 · Subcomando `qc`: frecuencia de muestreo, duración, offset DC y saturación · orquestador · los tests con señales sintéticas (offset conocido, recorte conocido) pasan; `qc` escribe un CSV en `outputs/qc/`
- [ ] F1-T06 · Manifiesto de partición train/val/test en `datasets/manifests/` · data-engineer · determinista (semilla fijada), estratificado por clase y versionado; dos ejecuciones dan el mismo fichero
- [ ] F1-T07 · Comprobación de fugas en `tools/` · data-engineer · falla si un buque o una grabación aparece en dos particiones; tiene un test con un caso con fuga y otro sin ella
- [ ] F1-T08 · Pasada del reviewer sobre F1 · reviewer · ningún hallazgo crítico ni alto abierto

## F2 · Señal

- **Objetivo:** preprocesado (quitar el offset DC, normalizar, segmentar), LOFAR y DEMON.
- **Entregable:** subcomando `features` y figuras de ejemplo en `outputs/`.
- **Criterio de salida:** tests con señales sintéticas: un tono conocido aparece en el LOFAR en
  su frecuencia, y una modulación conocida aparece en el DEMON en su frecuencia.

## F3 · Modelo

- **Objetivo:** una línea base clásica (rasgos + regresión logística o random forest) frente a
  una CNN pequeña en PyTorch.
- **Entregable:** subcomandos `train` y `evaluate`, y pesos exportados a `models/`.
- **Criterio de salida:** F1 macro, matriz de confusión e intervalos por bootstrap sobre test
  para los dos modelos; `evaluate` funciona en CPU con los pesos versionados.

## F4 · Entrega

- **Objetivo:** que el amigo pueda evaluar, y reentrenar si quiere, siguiendo solo el README.
- **Entregable:** README completo, informe de resultados (subcomando `report`) y etiqueta `v0.1`.
- **Criterio de salida:** la pasada del reviewer y la del friend-tester no dejan hallazgos
  bloqueantes.
