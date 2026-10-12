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

## 2026-10-12 · Proyecto cerrado: F5 y F6 completas, etiqueta `v0.2`
- **Qué se hizo:** F6-T02 (cierre): arreglados los hallazgos del reviewer final (procedencia de
  figuras e informe con el código de `9d72af5`, restos de `docs/results/README.md`, ECE 0,27,
  explicación de por qué difieren las dos cifras de variación, duración de los tests,
  `evaluate --check`/`--details` sin `--round 3` dan error con test, textos de ayuda). Friend-tester
  final en limpio, en CPU, sobre `9778de9`: sin atascos; tests 349/11 sin datos y 356/4 con datos
  como dice la tabla; `--check` reproduce (idéntico en 6 modelos, +0,0045 en el híbrido con
  aumento); `report` idéntico byte a byte, PNG incluidos; 59 enlaces e imágenes y todas las anclas
  correctas. Aplicados sus retoques de texto (cifras de `--check` de una sola partición, `wd`,
  "pruebas previas", duración de los tests, "algo más del doble"). F5-T09 y F6 cerradas; etiqueta
  `v0.2` en el estado final (David: cierre de todo el proyecto). David acepta su nombre en el ©
  de la licencia y en la URL de clonado.
- **Qué quedó a medias:** nada. Fuera de alcance, apuntado para un futuro proyecto: ROC de
  detección con un dataset con grabaciones sin barco (DT-007), más buques y clases grabadas en
  las mismas condiciones, y las opciones de velocidad comentadas con David (varios entrenamientos
  a la vez en la GPU y sacar del bucle el trabajo de CPU).
- **Ficheros tocados:** `README.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `git tag` (`v0.2`); `docker compose run --rm --entrypoint python pipeline
  -m pytest -rs`; `docker compose run --rm pipeline evaluate --round 3 --check`.
- **Siguiente paso exacto:** ninguno; si se retoma, abrir una fase nueva con David.

## 2026-10-12 · F5-T11 y F6-T01 cerradas: README limpio para terceros
- **Qué se hizo:** F5-T11 (`9d72af5`): README reescrito sin cronología ni rondas anteriores
  (resumen, datos y señal, protocolo con sus 9 referencias, modelos con las 3 del calentamiento,
  configuraciones, 15 gráficas con "cómo se lee" y "qué se ve", conclusiones, puesta en marcha,
  limitaciones, procedencia), `evaluate --round 3 --check` (7 modelos representativos en CPU,
  ~5 s, margen ±0,01), figuras de datos sin partición fija (`data_classes.png`,
  `data_partitions.png`), instrucciones de las rondas 1 y 2 movidas a `docs/results/README.md`,
  arreglos bajos del reviewer de F5-T10. Friend-tester en limpio en CPU: sin atascos, todo
  reproducido. F6 (nueva fase, David): F6-T01 aplica sus dudas: sin el nombre de David en el
  texto (salvo el © de la licencia, que David acepta), explicación de que DeepShip siempre tiene
  barco y de la ROC de detección frente a la de clasificación, F1 macro definido en el resumen,
  partición definida y los tres `.npz` explicados como almacenamiento (`features` ya no imprime
  la tabla train/val/test), `flag_dc` explicado (en las 63), sin códigos D-0xx/DT-0xx en el
  README, las dos cifras de variación con nombre propio, ayuda de la CLI sin fases internas y con
  el tiempo real de `--check`. Los comandos no cambian (`--round 3`, decisión de David).
- **Qué quedó a medias:** cierre de F5 (F5-T09) y etiqueta `v0.2`: falta que David decida si la
  etiqueta va antes o después de F6-T01; cierre de F6 (reviewer).
- **Ficheros tocados:** `README.md`, `src/sonar_pipeline/__main__.py`, `PLAN.md`, `HANDOFF.md`
  (F6-T01); F5-T11 en su commit.
- **Cómo verificarlo:** `docker compose run --rm --entrypoint python pipeline -m pytest -rs`
  (355 pasan, 4 saltados de CUDA); `docker compose run --rm pipeline evaluate --round 3 --check`.
- **Siguiente paso exacto:** preguntar a David por `v0.2` y por el cierre de F6.

## 2026-10-12 · F5-T10 cerrada: ronda 3 reentrenada con el lr arreglado
- **Qué se hizo:** código en `7df7a96`: el lr pasa al 100 % en la 6.ª época y la regla de meseta
  empieza entonces (test que lo exige); cada época guarda la norma del gradiente (media y
  máximo) y el tiempo de formar minibatches, y cada entrenamiento sus tiempos de carga de datos;
  `round3.json` registra los 4 hilos reales y el inicio real de la ronda; los JSON de
  interpretabilidad y detalles guardan dispositivo y hardware; el informe compara inicial y
  reentrenado. Con el visto bueno de David (aviso previo de uso de GPU), reentrenados los 7
  modelos (15:03-16:44, 1 h 41 min, RTX 4060), más interpretabilidad y detalles (~11 min).
  **La línea base y la CNN dan los mismos pesos bit a bit** (la instrumentación no cambia el
  entrenamiento). Las 4 redes afectadas eligen la misma configuración. Test reentrenado:
  híbrido 0,336 (antes 0,343), híbrido + aum. 0,304 (0,340), transformer 0,239 (0,235),
  transformer + aum. 0,249 (0,240). Ninguna de las 13 comparaciones es significativa (la más
  cercana, transformer frente a línea base, p 0,075, Holm 0,97). Resultados iniciales en
  `docs/results/round3_initial/` (pesos y JSON por entrenamiento iniciales en
  `outputs/round3_initial_run/`, no versionados). README (aviso de reentrenamiento y cifras),
  informe, 15 figuras (curvas con norma del gradiente; velocidad con el tiempo de minibatches),
  procedencia y D-024 (ajuste 3 cumplido) actualizados. Nuevo: el aumento de datos llega al 26 %
  del tiempo de entrenamiento del transformer; la norma del gradiente del híbrido con aumento
  crece con las épocas.
- **Qué quedó a medias:** F5-T09 (cierre): pasada corta del reviewer sobre F5-T10 y etiqueta
  `v0.2`.
- **Ficheros tocados:** `src/sonar_pipeline/{cv,round3_details,figures_round3,report,report_round3,interpret_cv,__main__}.py`,
  tests, `docs/results/`, `docs/figures/round3/`, `docs/report.md`,
  `models/round3/representative/`, `README.md`, `DECISIONS.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm --entrypoint python pipeline -m pytest -rs`
  (352 pasan, 4 saltados de CUDA); `docker compose run --rm pipeline report` sin cambios.
- **Siguiente paso exacto:** reviewer sobre F5-T10, hallazgos a David, cerrar F5 y etiquetar
  `v0.2`.

## 2026-10-12 · F5-T09: revisión pasada; fallo del lr declarado; F5-T10 obligatoria
- **Qué se hizo:** reviewer y friend-tester sobre `a6c9aac`. El friend-tester siguió el README en
  limpio en CPU sin atascos (cifras y reproducción idénticas; `report` sin cambios). El reviewer
  no encontró fugas, secretos ni audio, pero sí un **fallo alto**: el calentamiento del lr del
  transformer y del híbrido (sin y con aumento) se quedaba en el 82 % (4,1e-4 en lugar de 5e-4)
  tras la época 5 (`cv.py`, bucle de `fit_network`); la CNN y la línea base no se ven afectadas.
  Con David: se declara ya (ajuste 3 de D-024, aprobado, que recoge además las 13 comparaciones,
  la semilla 0 reutilizada y la semántica de ReduceLROnPlateau) y un aviso de resultados
  provisionales en el README; **F5-T10 pasa a obligatoria**: arreglar el lr, guardar la norma del
  gradiente y el tiempo de carga de datos, y reentrenar; los resultados reentrenados serán los de
  la ronda 3 y los actuales, histórico; el reentrenamiento se declara como la ronda 2. **`v0.2`
  se etiqueta después de F5-T10.** README ampliado: aviso, por qué hay calentamiento solo en
  transformer e híbrido (Liu et al. 2020, Vaswani et al. 2017, Ioffe y Szegedy 2015), qué
  modelos intervienen en cada sección, tabla de tests saltados, tiempo CPU frente a GPU, y las
  propuestas del friend-tester (rutas largas en Windows, qué regenera `report`, MFLOPs, ECE) y
  los arreglos de documentación del reviewer. **David pide aviso antes de cualquier uso de la
  GPU** para no usar el PC (memoria actualizada).
- **Qué quedó a medias:** 3 cambios de código ya revisados con David, sin commit, para F5-T10:
  ayuda de `--measure` (`__main__.py`), dispositivo, hardware y etiqueta `split` en
  `interpret_cv.py` y dispositivo y hardware en `round3_details.py`. El aviso de reentrenamiento
  en el informe va en F5-T10 (el informe se genera desde el código).
- **Ficheros tocados:** `README.md`, `DECISIONS.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** leer el ajuste 3 de D-024 y el aviso al principio del README.
- **Siguiente paso exacto:** F5-T10 según el plan aprobado (B1-B13): código y tests sin GPU;
  después avisar a David con el tiempo medido antes de usar la GPU.

## 2026-10-12 · F5-T08 cerrada: gráficas, informe y README de la ronda 3
- **Qué se hizo:** `evaluate --round 3 --details` (`round3_details.py`, 1 min en GPU, sin
  entrenar) reúne en `docs/results/round3_details.json` lo que las gráficas necesitaban y solo
  estaba en `outputs/`: velocidad, curvas, F1 de entrenamiento y validación por inferencia,
  variación por partición y por semilla, acierto por buque, acuerdo entre modelos, calibración
  (ECE, Brier, log loss) y ROC. `figures_round3.py` dibuja las 15 gráficas acordadas en
  `docs/figures/round3/`, y `report_round3.py` añade al informe el capítulo de la ronda 3
  (cifras leídas de los JSON) y las rondas 1 y 2 como histórico. README reorganizado por etapas:
  resumen con el resultado de la ronda 3, cronología, índice, uso común, cómo reproducir la ronda
  3, datos y señal, ronda 3 (protocolo con literatura, **arquitecturas explicadas**: CNN,
  transformer e híbrido con BatchNorm, LeakyReLU, RMSNorm, RoPE, GQA, SwiGLU, AdamW, *weight
  decay*, calentamiento, decaimiento del lr, parada temprana y aumento; **justificación de cada
  configuración** con la hipótesis que prueba y su F1 de validación; las 5 gráficas principales
  explicadas, con "4. Resultados en test" como diagrama de caja), rondas 1 y 2 como histórico
  no comparable, limitaciones y procedencia. `evaluate` admite `--outputs-dir`. Nuevo dato: la
  línea base acierta el 100 % de entrenamiento y es la peor calibrada (ECE 0,45). No se
  guardaron la norma del gradiente ni los tiempos de carga de datos que se mencionaron al
  planificar: no se pueden recuperar sin reentrenar.
- **Qué quedó a medias:** nada de F5-T08.
- **Ficheros tocados:** `src/sonar_pipeline/{round3_details,figures_round3,report_round3,report,__main__}.py`,
  `tests/test_report_round3.py`, `docs/results/round3_details.json`, `docs/results/README.md`,
  `docs/figures/round3/`, `docs/report.md`, `README.md`, `PLAN.md`, `HANDOFF.md`.
- **Cómo verificarlo:** `docker compose run --rm --entrypoint python pipeline -m pytest -rs`
  (347 pasan y 4 se saltan en CPU; 350 y 1 en GPU); `docker compose run --rm pipeline report`
  no cambia `docs/report.md`.
- **Siguiente paso exacto:** F5-T09: reviewer, friend-tester con el README nuevo, cierre de F5
  y etiqueta `v0.2`.
