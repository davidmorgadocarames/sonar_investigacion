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

## 2026-10-12 · F5 empezada; F5-T01 cerrada: protocolo de la ronda 3 (D-024)
- **Qué se hizo:** David aprobó F5 (9 tareas, sin ARM ni preentrenados) y D-024: validación
  cruzada repetida 5 × 5 por buque (60/20/20, rotando entrenamiento, validación y test; cada buque
  en test 5 veces), 7 modelos (línea base; CNN, transformer puro e híbrido, sin y con aumento),
  4 configuraciones A-D por tipo de red elegidas con la media de validación y 1 semilla, modelo
  final con 3 semillas promediadas, técnicas (AdamW, calentamiento, RMSNorm, RoPE, SwiGLU, GQA),
  prueba de Nadeau y Bengio con Holm, JSON por entrenamiento (MLOps, MFLOPs, tokens/s...), pesos
  de F5 sin versionar, reproducción por el amigo opcional y solo GPU. README: sección "Ronda 3
  (F5, en curso): protocolo de evaluación" con 9 referencias verificadas; "En gráficas" titulada
  como rondas 1 y 2. La reorganización completa por etapas del README va en F5-T08.
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `DECISIONS.md`, `README.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** leer D-024 y la sección del protocolo en el README.
- **Siguiente paso exacto:** F5-T02: particiones 5 × 5 con el subagente data-engineer.

## 2026-10-11 · README con las gráficas explicadas (`v0.1.1`)
- **Qué se hizo:** a petición de David, la sección "En gráficas" del README explica cada figura:
  qué son Cargo, Passengership, Tanker y Tug, y entrenamiento, validación y test (y por qué se
  parte por buque); qué son LOFAR y DEMON y qué prueban los 440 Hz y 12 Hz sintéticos; cómo leer
  el LOFARgram y el DEMON del MALASPINA (líneas tonales hacia 80, 594 y 682 Hz; eje hacia
  2,8-3 Hz; paso de pala cada 11,3 Hz y armónicos, que sugieren 4 palas; cifras comprobadas con
  los rasgos); los tres modelos, las dos rondas y cómo leer el F1 con intervalos y azar; las
  matrices de confusión y por qué las CNN colapsan hacia una clase (explicación razonada, no
  demostrada); y qué mide el eje vertical de cada panel de bandas. Solo el README (no el
  informe). Etiqueta `v0.1.1` sobre este commit (`v0.1` no se mueve, regla 9).
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `README.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** ver el README en GitHub en la etiqueta `v0.1.1`.
- **Siguiente paso exacto:** ninguno obligatorio; F5 (opcional) si David la pide.

## 2026-10-11 · F4 Entrega cerrada (F4-T07) y etiqueta `v0.1`
- **Qué se hizo:** pasada del reviewer sobre F4: sin hallazgos críticos ni altos; 5 medios y 11
  bajos, todos resueltos con la aprobación de David. Informe: conclusiones calculadas desde los
  JSON (por encima o por debajo del azar solo si todo el intervalo lo está), p de la validación
  cruzada separado de test, DT-005 con la cifra de la duración (0,93 / 0,80), CNN con aumento a
  0,071 por grabación en val, Pd interpolada en la ronda 1; tests de la rama de la ronda 1 y de
  que `docs/report.md` está al día. README: sección "En gráficas" con las figuras principales,
  "Equipo de referencia" (i7-14700KF, RTX 4060 8 GB, driver 610.62, Windows 10 + Docker Desktop
  WSL2), reproducción exacta comprobada en x86-64, tiempos aclarados, `.env` de Linux con
  `printf`, Docker sin root y Podman con `HOST_UID=0` (comprobado que la imagen funciona con el
  UID 0), `tools/check_reproduction.py` (±0,005; tu `metrics.json` lo pasa) y licencia. Tabla de
  procedencia: código de `v0.1` con `code_sha256` `b4b6638e…`, filas de las figuras de la ronda 1
  y del informe; lo cambiado en F4 no afecta a ninguna cifra (comprobado con git). D-022 (MIT, ©
  2026 David Morgado Carames; no cubre datos ni pesos) y D-023 (reproducción fuera de x86-64 y
  Docker sin root); la prueba ARM emulada queda en F5. CLAUDE.md con el volumen `docs/`, ruta
  genérica en PLAN, código muerto quitado. 251 tests.
- **Qué quedó a medias:** nada de F4. F5 (mejoras opcionales: atención, sobreajuste de las CNN,
  prueba en ARM) solo si David la pide.
- **Ficheros tocados:** `LICENSE`, `README.md`, `CLAUDE.md`, `docs/report.md`,
  `docs/results/README.md`, `src/sonar_pipeline/{report,__main__}.py`,
  `tools/check_reproduction.py`, `tests/test_{report,check_reproduction}.py`, `DECISIONS.md`,
  `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `git show v0.1`; en un clon limpio, seguir el README y ejecutar
  `tools/check_reproduction.py`.
- **Siguiente paso exacto:** ninguno obligatorio. Si David quiere seguir, detallar F5 con el
  procedimiento de inicio de fase.

## 2026-10-11 · F4-T06 cerrada: el friend-tester reproduce la evaluación 2
- **Qué se hizo:** con el visto bueno de David, el friend-tester clonó `b581d0c` en una carpeta
  temporal (ya borrada) y siguió solo el README por la opción A (CPU), sin entrenar: build
  1 min 14 s (con caché), tests sin datos 237 passed y 8 skipped, descarga de DeepShip ~1 min,
  `qc` 4 s, `features` 17 s, tests con datos 244 passed y 1 skipped, **`evaluate` 30 s con cifras
  idénticas a la tabla del README y a `docs/results/eval2.json`** (F1, intervalos y
  `weights_sha256` de los tres modelos), `report` 2 s con `git status` vacío (informe idéntico).
  Espacio: clon 21 MB, DeepShip 1,2 GB, rasgos 82 MB. Su `code_sha256` (`5abdb24b…`) corresponde a
  `b581d0c`. Con la aprobación de David: README con el tercer motivo de tests saltados, `flag_dc`
  esperables y aviso de que el `code_sha256` difiere; tabla de procedencia con la reproducción y
  el "no comprobado" de los hilos pasado a comprobado (cifras idénticas con 4 hilos).
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `README.md`, `docs/results/README.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** repetir los pasos del README en un clon limpio.
- **Siguiente paso exacto:** F4-T07: reviewer, cierre de F4 y etiqueta `v0.1`.

## 2026-10-11 · F4-T05 cerrada: DT-001 pagada (UID y GID del host)
- **Qué se hizo:** `compose.yaml` ejecuta el contenedor con `user: "${HOST_UID:-1000}:${HOST_GID:-1000}"`
  (en Linux, en `.env` con `id -u` e `id -g`; en Windows y macOS no hace falta). La imagen usa
  `HOME=/tmp`, `MPLCONFIGDIR=/tmp/matplotlib` y pytest sin caché, porque un UID del host no tiene
  carpeta personal ni puede escribir en `/app`. Comprobado con el UID 4321: 244 tests en verde y
  `report` escribe en una carpeta nueva ficheros a su nombre (informe idéntico). La prueba también
  reprodujo el problema de DT-001: con un UID distinto no se pueden sobrescribir ficheros creados
  por otro UID. La imagen GPU sigue viendo la RTX 4060. `tests/test_container.py` (2 tests).
  README: paso "Solo en Linux". D-019 con el ajuste del separador (aprobado por David).
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `Dockerfile`, `compose.yaml`, `tests/test_container.py`, `README.md`,
  `DECISIONS.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `HOST_UID=4321 HOST_GID=4321 docker compose run --rm --entrypoint python
  pipeline -m pytest -q`.
- **Siguiente paso exacto:** F4-T06: pasada del friend-tester (clon en la carpeta temporal
  aprobada; opción A; debe reproducir `docs/results/eval2.json`).
