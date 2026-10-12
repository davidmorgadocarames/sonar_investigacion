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

## 2026-10-12 · F5-T03 en curso: plan para hacer T03, T04 y T05 seguidos
- **Qué se hizo:** David pidió hacer F5-T03 (motor de validación cruzada), F5-T04 (transformer)
  y F5-T05 (híbrido) seguidos, y que **todas las preguntas sobre la ronda 3 se le hagan antes**,
  para que después la ronda 3 corra con un solo comando sin acciones suyas (quiere dejarla
  entrenando de noche). Diseño en D-024 y su ajuste 1; particiones en
  `datasets/manifests/cv_splits.csv`.
- **Qué quedó a medias:** F5-T03 recién empezada (sin código aún). Preguntas a David pendientes
  de respuesta: autorización para lanzar la ronda 3 sin esperarle si los tests pasan y la
  estimación medida es ≤ 4 h de GPU; qué hacer si un bloque falla; commit y push automáticos de
  los resultados.
- **Ficheros tocados:** `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** PLAN.md (F5-T03 en `[~]`).
- **Siguiente paso exacto:** escribir el motor de F5-T03 (`src/sonar_pipeline/cv.py`: entrena
  cualquier modelo en las 25 particiones, elige con media de validación y semilla 0, final con
  semillas 0-2 promediadas, JSON por entrenamiento con MLOps, MFLOPs, tokens/s, tiempos, curvas,
  calibración, predicciones; reanudable; prueba de Nadeau y Bengio con Holm) y sus tests
  sintéticos; luego F5-T04 y F5-T05; un comando único `train --round 3`.

## 2026-10-12 · F5-T02 cerrada: particiones 5 × 5 por buque
- **Qué se hizo:** `data-engineer` creó `tools/make_cv_splits.py` y
  `datasets/manifests/cv_splits.csv` (1575 filas, md5 `3558f6ff…`): 25 particiones (5
  repeticiones × 5 vueltas). Clases con 5 o más buques: 5 folds por repetición con minutos
  equilibrados (tolerancia sobre el mejor reparto), vuelta k con test = fold k y validación =
  fold k + 1; cada buque una vez en test y una en validación por repetición. Tras una primera
  versión en la que Tug faltaba de test en 10 particiones y de validación en otras 10, David
  aprobó que las clases con menos de 5 buques vayan 1/1/resto: Tug 1/1/1 en las 25 (test 9/8/8,
  validación 8/9/8) y la métrica simplificada (F1 macro de 4 clases por partición). Ajuste 1 de
  D-024 con estas reglas, el algoritmo con tolerancia, el desequilibrio de minutos (Cargo 25 %,
  Passengership 42 %, Tanker 10,5 %) y la dependencia del generador de numpy. Sin fugas en
  ninguna partición; 4 clases en los tres papeles en todas; regeneración idéntica. 41 tests
  nuevos (288 en total).
- **Qué quedó a medias:** nada.
- **Ficheros tocados:** `tools/make_cv_splits.py`, `datasets/manifests/cv_splits.csv`,
  `tests/test_make_cv_splits.py`, `tests/test_cv_splits_data.py`, `datasets/README.md`,
  `DECISIONS.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm --entrypoint python pipeline
  tools/make_cv_splits.py --summary` y `git status` sin cambios.
- **Siguiente paso exacto:** F5-T03: motor de validación cruzada y JSON por entrenamiento.

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
