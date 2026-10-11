# PLAN.md

**Única fuente de estado del proyecto.** Solo se detalla en tareas la fase actual; las demás
quedan a alto nivel.

- **Formato de tarea:** `- [ ] F1-T01 · descripción · responsable · criterio de aceptación verificable`
- **Estados:** `[ ]` pendiente · `[~]` en curso · `[x]` hecho · `[!]` bloqueado (con motivo)
- **Fase actual:** ninguna obligatoria: F0-F4 hechas (`v0.1`); F5 es opcional

| Fase | Nombre | Estado |
|---|---|---|
| F0 | Setup | [x] |
| F1 | Datos | [x] |
| F2 | Señal | [x] |
| F3 | Modelo | [x] |
| F4 | Entrega | [x] |
| F5 | Mejoras opcionales | [ ] |

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

## F1 · Datos — [x] hecha (2026-10-11)

- **Objetivo:** tener DeepShip localizado, verificado, con control de calidad y particionado sin
  fugas, y decidir cómo le llegan los datos al amigo.
- **Entregable:** comandos de descarga en el README (D-006), subcomando `qc`, manifiesto de partición
  versionado y comprobación de fugas.
- **Criterio de salida:** licencia revisada y supuesto aceptado por David (D-014); decisión de distribución aprobada; manifiesto en
  `datasets/manifests/`; la comprobación de fugas pasa; la pasada del reviewer no deja
  hallazgos graves abiertos.

- [x] F1-T01 · Confirmar la licencia y la fuente oficial de DeepShip · David · licencia y URL anotadas en `datasets/README.md`, sin "PENDIENTE DE VERIFICAR"
- [x] F1-T02 · Proponer cómo le llegan los datos al amigo (depende de F1-T01) · orquestador → aprobación de David · decisión aprobada y apuntada en DECISIONS.md (D-005 y D-006)
- [x] F1-T03 · Inventariar DeepShip: estructura de carpetas, metadatos y si identifican el buque · data-engineer · informe con conteos por clase, buque y grabación, y unidad de partición elegida (buque o grabación) con su justificación (resultado: 63 WAV, 44 buques, partición por buque, D-007)
- [x] F1-T04 · Comandos de descarga en el README (`git clone -c core.autocrlf=false` + checkout del commit fijado, D-006) · orquestador; prueba en limpio: friend-tester · siguiendo solo la sección "Datos" del README en una carpeta temporal limpia: quedan 63 WAV en `datasets/raw/DeepShip` (12/20/28/3), idénticos byte a byte a la fuente, y `git status` del proyecto no muestra ningún audio
- [x] F1-T05 · Subcomando `qc` con numpy y scipy (D-008): frecuencia de muestreo, mono, float32, duración, offset DC, saturación (\|x\| ≥ 1,0), NaN o infinitos y silencio · orquestador · tests con señales sintéticas de respuesta conocida que pasan; `qc` escribe `outputs/qc/qc.csv` con una fila por WAV y no excluye ningún fichero (la comparación de duración con el metafile se hace en F1-T06b)
- [x] F1-T05b · Proponer qué hacer con las grabaciones que el QC marque (muy cortas, con offset DC, saturadas...): excluirlas, conservarlas o tratarlas aparte · orquestador → David · decisión aprobada y apuntada en DECISIONS.md (D-009)
- [x] F1-T06a · Tabla WAV → buque en `datasets/manifests/`, generada por un script de `tools/` a partir de los metafiles (D-007) · data-engineer · incluye el SHA-256 de cada WAV y el origen de cada asignación (id o duración); los grupos dudosos van juntos; al regenerarla sale el mismo fichero
- [x] F1-T06b · Comprobación de fugas en `tools/` y test de DT-002, antes de partir · data-engineer · falla si un buque o una grabación aparece en dos particiones; tiene un test con un caso con fuga y otro sin ella; otro test comprueba que la duración de cada WAV coincide con la de su fila del metafile en los 61 casos verificados (paga DT-002); un listado de grupos con nombres parecidos revisado con David (paga DT-003)
- [x] F1-T06c · Partición train/val/test por buque en `datasets/manifests/` · data-engineer, con las proporciones aprobadas por David (D-013) · determinista (búsqueda exacta sin azar, D-013); un buque de Tug en cada partición; reparto equilibrado por duración; la comprobación de fugas de F1-T06b pasa y queda como test permanente sobre el `split.csv` versionado (D-012); al regenerarla sale el mismo fichero
- [x] F1-T08 · Pasada del reviewer sobre F1 y cierre de fase · reviewer + orquestador · hallazgos enseñados a David; ningún hallazgo crítico ni alto abierto; decisiones y deudas propuestas

## F2 · Señal — [x] hecha (2026-10-11)

- **Objetivo:** preprocesado (quitar el offset DC, normalizar, segmentar), LOFAR y DEMON.
- **Entregable:** subcomando `features` y figuras de ejemplo en `outputs/`.
- **Criterio de salida:** tests con señales sintéticas: un tono conocido aparece en el LOFAR en
  su frecuencia, y una modulación conocida aparece en el DEMON en su frecuencia; la pasada del
  reviewer no deja hallazgos graves abiertos.

- [x] F2-T01 · Pagar DT-004: test que regenera `vessels.csv` · data-engineer · el test ejecuta `build_vessel_table.py` sobre `datasets/raw/DeepShip`, escribe en una carpeta temporal y compara byte a byte con la versionada; se salta si no hay audios; pasa en Docker
- [x] F2-T02 · Pagar DT-006 (mejoras menores) · orquestador (+ data-engineer para `tools/`) · id repetido avisa o falla; `qc.csv` en LF y ordenado; el test de `qc` compara contenido; el test de `--help` comprueba los subcomandos de verdad; imagen base con versión exacta; `.gitignore` sin distinguir mayúsculas; los tests de datos se saltan si no hay audios; el comando de tests de CLAUDE.md lleva `-rs` para que se vean los saltados (aprobado por David; el del README queda para F4 porque el README aún no tiene comando de tests); cada punto con su test o verificación
- [x] F2-T03 · Proponer parámetros de preprocesado, LOFAR, DEMON y formato de salida · orquestador → David · decisión aprobada y apuntada (D-015); si el segmento dura más de 6 s, revisar D-009
- [x] F2-T04 · Módulo de preprocesado: offset DC por grabación (D-009), normalización y segmentación · orquestador · tests sintéticos: el offset desaparece, número de segmentos exacto, ningún segmento cruza grabaciones, cada grabación da al menos un segmento
- [x] F2-T05 · LOFAR (STFT, dB, normalización de fondo) · orquestador · un tono sintético de frecuencia conocida da el máximo en su bin (±1 bin); el ruido blanco normalizado queda plano
- [x] F2-T06 · DEMON (paso banda, envolvente con Hilbert, paso bajo, diezmado, espectro) · orquestador · ruido en banda modulado a `fm` conocida da un pico en `fm` (±1 bin); sin modulación no hay pico
- [x] F2-T07 · Subcomando `features` · orquestador · lee `split.csv`, escribe en `datasets/processed/` con un índice por segmento (`wav_path`, clase, partición, n.º de segmento); determinista; test de principio a fin con un mini dataset sintético; sobre los datos reales informa de los segmentos por clase y partición; README actualizado
- [x] F2-T08 · Figuras de ejemplo en `outputs/figures/` (LOFARgram y DEMON de una grabación por clase, y validación sintética) · orquestador · se generan desde la CLI; matplotlib aprobada antes como dependencia (regla 3, D-002)
- [x] F2-T09 · Pasada del reviewer y cierre de F2 · reviewer + orquestador · hallazgos enseñados a David; ningún hallazgo crítico ni alto abierto; decisiones y deudas propuestas
- [x] F2-T10 · Arreglar los hallazgos de la revisión de F2 (aprobado por David) · orquestador · LOFAR diezmado a 5 kHz sin aliasing en 0-2 kHz (ajuste de D-015) con test de tonos a 2050 y 2100 Hz; todas las dependencias fijadas (D-017); `features` comprueba el SHA-256 de los WAV, guarda el SHA de `vessels.csv` y el commit de DeepShip y borra `.npz` antiguos; figuras solo de train y ejes correctos; tests de bin exacto, aliasing en DEMON, independencia entre grabaciones y partición del índice; documentación coherente

## F3 · Modelo — [x] hecha (2026-10-11)

- **Objetivo:** una línea base clásica (regresión logística sobre rasgos) frente a una CNN pequeña
  en PyTorch, entrenada sin y con aumento de datos.
- **Entregable:** subcomandos `train` y `evaluate`, pesos en `models/` y
  `outputs/results/metrics.json` con figuras.
- **Reglas de la fase (David, 2026-10-11):**
  - **Test no se mira hasta F3-T09.** Todo ajuste y toda comparación (también con o sin aumento
    de datos) se decide con val o con validación cruzada dentro de train + val.
  - **Avisar a David y esperar su visto bueno antes de lanzar cualquier entrenamiento o test
    de permutación, con una estimación del tiempo que tardará** (para que no use el PC).
  - Primero se hace commit de un modelo entrenado y después se lanza el test de permutación.
  - Los pesos se guardan siempre en `models/` (volumen montado, quedan en el PC de David); si
    ocupan menos de 10 MB se versionan también en git.
- **Contexto (notas de F1 y F2):** Tug tiene 3 buques (1/1/1), así que su métrica en test sale de
  una sola grabación; cada clase aporta un número de segmentos muy distinto (D-009), compensado
  con pesos por clase; en el total test pesa un 23 % (57/20/23) y en Tanker el reparto con coste 0
  se eligió en orden fijo (D-013); los intervalos se calculan remuestreando grabaciones, no
  segmentos.
- **Descartado (David, 2026-10-11):** SwiGLU, RoPE, RMSNorm, Grouped Query Attention (no aportan
  a una CNN con estos datos) y SHAP (dependencias pesadas; se usa importancia por permutación).
- **Criterio de salida:** F1 macro, matriz de confusión e intervalos por bootstrap sobre test para
  la línea base, la CNN y la CNN con aumento; `evaluate` funciona en CPU con los pesos
  versionados; la pasada del reviewer no deja hallazgos graves abiertos.

- [x] F3-T01 · Proponer el diseño del modelo y de la evaluación · orquestador → David · decisión aprobada y apuntada (D-018): línea base, arquitectura y entrenamiento de la CNN, aumento de datos, agregación por grabación, validación cruzada y test de permutación por buque, contenido de `metrics.json`, interpretabilidad, versionado de pesos
- [x] F3-T02 · Dependencias: scikit-learn y PyTorch CPU · orquestador · decisión aprobada (D-019); PyTorch desde `TORCH_INDEX_URL` en un paso aparte; lista completa regenerada (D-017); build OK y tamaño de la imagen medido e informado
- [x] F3-T03 · Módulo de datos: carga de `.npz` e índice, etiquetas, pesos por clase y agregación por grabación (media de probabilidades) · orquestador · tests con datos de juguete; nunca mezcla particiones
- [x] F3-T04 · Módulo de métricas y `metrics.json` · orquestador · F1 macro y por clase con bootstrap por grabaciones; matriz de confusión por segmento y por grabación; ROC Pd/Pfa una frente al resto con AUC y Pd a Pfa = 1 %; trazabilidad (SHA-256 de `split.csv` y de los rasgos, `code_sha256`, semillas, versiones); parámetros, FLOPs, tiempo de entrenamiento y de inferencia con el hardware y los hilos; tests con casos de resultado conocido
- [x] F3-T05 · Comprobar DT-005 con un clasificador trivial (solo duración de la grabación) · orquestador · F1 en val informado a David y decisión sobre si preocupa
- [x] F3-T06 · Línea base: regresión logística (`train --model baseline`) · orquestador · estandarización y pesos por clase; hiperparámetros con val; validación cruzada agrupada por buque en train + val; pesos en `models/` con commit; después, test de permutación con etiquetas permutadas por buque; **aviso a David con estimación de tiempo antes de entrenar y antes de permutar**
- [x] F3-T07 · CNN base sin aumento de datos (`train --model cnn`) · orquestador · rama 2D (LOFAR) y 1D (DEMON); BatchNorm, LeakyReLU, AdamW con weight decay, decaimiento del lr, minibatches de 32-64, pesos por clase, parada temprana con F1 macro de val, semillas fijadas; pesos en `models/`; **aviso a David con estimación de tiempo antes de entrenar**
- [x] F3-T08 · CNN con aumento de datos (`train --model cnn-aug`) · orquestador · misma CNN y semillas, con enmascarado de frecuencia y tiempo y pequeños desplazamientos solo en train; comparación con F3-T07 en val; se conservan e informan los dos modelos; **aviso a David con estimación de tiempo antes de entrenar**
- [x] F3-T09 · Subcomando `evaluate` · orquestador · evalúa test una sola vez con los tres modelos; escribe `outputs/results/metrics.json` (con la métrica general de 4 clases y la de dentro de cada época, y el clasificador de duración de DT-005 como referencia) y las figuras (matriz de confusión, F1 con intervalos, ROC Pd/Pfa); funciona en CPU con los pesos versionados
- [x] F3-T10 · Interpretabilidad · orquestador · línea base: importancia por permutación de bandas de frecuencia; CNN: saliencia por gradientes y oclusión sobre el LOFARgram; mapas en `.npz` y PNG, resumen por banda y clase en `metrics.json`; comprobar si las bandas importantes son líneas fijas compartidas por todos los buques de una época (50/60 Hz, armónicos del equipo), señal de que se aprende la época (DT-005)
- [x] F3-T11 · Pasada del reviewer y cierre de F3 · reviewer + orquestador · hallazgos enseñados a David; ningún hallazgo crítico ni alto abierto; decisiones y deudas propuestas
- [x] F3-T12 · Arreglar los hallazgos de la revisión de F3 y reentrenar (aprobado por David; ajuste 2 de D-018) · orquestador · pooling sin ceros con test de formas reales; hilos fijados; generadores separados; ReduceLROnPlateau; Pd a Pfa = 1 % no medible si hay pocos negativos y ROC de clasificación; hash de código y hardware de entrenamiento; permutación con C dentro; azar medido; tests de folds por buque, interpretabilidad sin test y determinismo en CUDA; figuras desde `metrics.json`; reentrenar las dos CNN (y la línea base solo para trazabilidad) y repetir la permutación, **con aviso y estimación de tiempo a David antes de cada paso**; segunda y última evaluación de test, conservando la primera en `outputs/results/eval1/`

## F4 · Entrega — [x] hecha (2026-10-11, etiqueta `v0.1`)

- **Objetivo:** que el amigo pueda evaluar, y reentrenar si quiere, siguiendo solo el README.
- **Entregable:** README completo, informe de resultados (subcomando `report`) y etiqueta `v0.1`.
- **Nota:** si el amigo no ha descargado los audios, los tests que los necesitan se saltan sin
  llamar la atención; revisarlo con el friend-tester.
- **Nota (David, 2026-10-11):** añadir al README una sección corta "Comprobar la instalación"
  con el comando de tests con `-rs` (`docker compose run --rm --entrypoint python pipeline -m
  pytest -rs`), para que se vean los tests saltados y su motivo.
- **Nota (DT-005, aprobada por David el 2026-10-11):** el informe declara la limitación junto a
  los resultados: en este subconjunto la clase coincide con la época (Cargo y Tug en 2017,
  Passengership y Tanker en 2016), así que el resultado de 4 clases no distingue el tipo de
  buque de las condiciones de grabación; solo las comparaciones dentro de una época están
  libres de ese efecto; Tug depende de una sola grabación en test.
- **Nota (revisión de F3, aprobada por David):** el informe muestra las dos evaluaciones de test
  (la primera y la de después de los arreglos de F3-T12) y explica por qué hubo una segunda;
  declara que Tug no es fiable y que las demás clases tienen pocos buques; la ROC es de
  clasificación (DT-007: la detección frente a ruido no se puede medir con DeepShip); el azar
  de referencia es el medido, no 0,25 fijo; la validación cruzada mezcla folds de 3 y 4 clases.
- **Nota (D-020):** los resultados de F3 de las dos rondas están resumidos y versionados en
  `docs/results/` (con la tabla de procedencia código ↔ resultado); el informe parte de ahí y
  sigue la regla de procedencia de D-020. Hechos que debe recoger: los de la sección final de
  `docs/results/README.md` (Tug no fiable, CNN con aumento por debajo del azar por grabación en
  val, C en el borde de la rejilla, CV con folds de 3 y 4 clases, ROC de clasificación).
- **Nota (propuesta de resultados):** el subcomando `report` genera la sección de resultados del
  README y las figuras solo a partir de `metrics.json`. Las figuras del README deben quedar
  versionadas fuera de `outputs/` (p. ej. `docs/figures/`), lo que requiere una decisión. Incluir
  también las gráficas de datos (minutos por clase y partición) y de señal (LOFARgram y DEMON de
  ejemplo por clase, y validación con señales sintéticas).
- **Criterio de salida:** la pasada del reviewer y la del friend-tester no dejan hallazgos
  bloqueantes.

- [x] F4-T01 · Decisiones de F4 · orquestador → David · D-021 aprobada: figuras en `docs/figures/` (las de la ronda 1, copiadas), informe `docs/report.md` generado por `report` desde lo versionado, DT-001 pagada ya con el UID del host, etiqueta `v0.1` tras el reviewer, sin virtualenv
- [x] F4-T02 · Figura de los datos: minutos por clase y partición · orquestador · sale de `vessels.csv` y `split.csv` sin audio; test
- [x] F4-T03 · Subcomando `report` · orquestador · genera `docs/report.md` (español) y copia las figuras a `docs/figures/` solo desde lo versionado (`docs/results/`); datos, señal, método, las dos rondas, limitaciones (DT-005, DT-007, Tug, pocos buques, azar medido, CV de 3 y 4 clases, dos evaluaciones) y procedencia; no entrena ni puntúa test; test que comprueba que las cifras del informe coinciden con los JSON y que al repetirlo sale igual
- [x] F4-T04 · README completo · orquestador · opciones A (CPU) y B (GPU con `.env`), "Comprobar la instalación" (`-rs`), datos, QC y rasgos, evaluar con los pesos entrenados, reentrenar (tiempos CPU y GPU), resultados en breve con enlace al informe, limitaciones, carpetas, licencia y cita; por qué no hace falta virtualenv; tiempos y espacio medidos
- [x] F4-T05 · Pagar DT-001: UID y GID del host en el compose (1000 por defecto) · orquestador · el contenedor escribe en los volúmenes con el usuario del host; matplotlib y pytest funcionan con un UID sin home; test o comprobación con un UID distinto de 1000
- [x] F4-T06 · Pasada del friend-tester (clon en `%TEMP%\tmp.XXXXXXXXXX\sonar_investigacion`, aprobado por David) · friend-tester · siguiendo solo el README por la opción A: build, descarga, `qc`, `features`, tests con `-rs` y `evaluate`; reproduce exactamente las cifras de `docs/results/eval2.json` (la reproducción se declara en la tabla de procedencia); tiempos, espacio y atascos
- [x] F4-T07 · Reviewer, cierre de F4 y etiqueta `v0.1` · reviewer + orquestador · sin hallazgos bloqueantes; etiqueta creada y publicada

## F5 · Mejoras opcionales — después de F4

- **Objetivo (David, 2026-10-11):** fase completamente opcional, al terminar el proyecto entero:
  comparar con la CNN base una capa pequeña que pondere las 11 tramas de tiempo y un transformer
  diminuto sobre trozos de frecuencia.
- **Nota:** con 43 buques el riesgo de sobreajuste es alto; la comparación se hace con las mismas
  reglas que F3 (val para decidir, test una sola vez).
- **Nota (David, 2026-10-11): reducir el sobreajuste de las CNN.** En F3 las dos CNN alcanzan su
  mejor val en las épocas 3-10 mientras la pérdida de train baja a 0,06-0,10, y el F1 de val
  varía mucho entre semillas (ronda 1: ±0,028 sin aumento y ±0,110 con aumento; ronda 2,
  tras F3-T12: ±0,069 y ±0,044; en la ronda 2 el mejor val llega en la época 3). Opciones a valorar: red más
  pequeña, más dropout o weight decay, lr más bajo, más aumento de datos, promediar semillas.
  Cualquier cambio se aplica igual a las dos CNN (mismo weight decay y misma curva de lr) para
  que la comparación siga siendo justa. **Test ya se habrá mirado en F3-T09**, así que el
  informe debe declarar que estas mejoras se hicieron después.
- **Nota (D-023, David, 2026-10-11): reproducción en ARM.** Construir la imagen para
  `linux/arm64` con `docker buildx` (emulación) y comprobar `evaluate` con
  `tools/check_reproduction.py`; puntúa test solo para comprobar la reproducción y se
  declara en la tabla de procedencia.
