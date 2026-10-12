# Resultados versionados · rondas 1, 2 (F3) y 3 (F5)

**La evaluación principal es la ronda 3** (validación cruzada 5 × 5 por buque, F5). Las rondas 1 y
2 (una partición fija, F3) se conservan como histórico: sus cifras no son directamente
comparables con las de la ronda 3.

## Ronda 3 (F5)

| Fichero | Qué contiene | Cómo se generó |
|---|---|---|
| [round3.json](round3.json) | Los 7 modelos: elección de configuración (validación de cada configuración en las 25 particiones), F1 de test por partición y semilla, dentro de cada época, matrices conjuntas, coste, tiempos y las 13 comparaciones con Nadeau y Bengio y Holm. | `train --round 3` (escrito al terminar cada bloque) |
| [round3_interpretability.json](round3_interpretability.json) | Oclusión, saliencia, permutación y atención por banda sobre validación, y la comprobación de época (DT-005). | `evaluate --round 3 --interpretability` |
| [round3_details.json](round3_details.json) | Lo que necesitan las gráficas y no está en los anteriores: velocidad, tiempos de carga de datos, curvas de aprendizaje con la norma del gradiente, F1 de entrenamiento y validación, acierto por buque, acuerdo entre modelos, calibración y ROC. | `evaluate --round 3 --details` (sin entrenar; test se lee de lo guardado por la ronda) |
| [round3_initial/](round3_initial/) | Los mismos tres ficheros de la **ronda 3 inicial**, antes del reentrenamiento de F5-T10 (histórico). | Copiados tal cual antes de reentrenar |

**Reentrenamiento (F5-T10, ajuste 3 de D-024).** En la ronda 3 inicial, el transformer y el
híbrido (sin y con aumento) entrenaron con un lr máximo de 4,1e-4 en lugar de 5e-4: el
calentamiento se quedaba en el 82 % al terminar. La revisión del código lo encontró; se arregló y
se reentrenó la ronda completa **después de haber visto test**, como en la ronda 2: un arreglo de
implementación decidido por la revisión, no por las cifras. La línea base y la CNN dan los mismos
pesos bit a bit; las cuatro redes afectadas eligen la misma configuración, y ninguna de las 13
comparaciones es significativa ni antes ni después. El informe compara las dos versiones.

- **Entrenamientos que no cambian:** si la mejor época cae dentro del calentamiento (épocas 1-5,
  idénticas antes y después), los pesos son los mismos que en la ronda inicial. Pasa en el 42 %
  de los entrenamientos del transformer, el 40 % del transformer con aumento, el 18 % del
  híbrido y el 17 % del híbrido con aumento, y en 3 de los modelos representativos
  (transformer, transformer con aumento e híbrido): por eso sus ficheros no cambian.
- **Tiempos:** la ronda reentrenada mide la norma del gradiente y sincroniza la GPU para medir
  el tiempo de cada minibatch; sus tiempos son algo mayores y no se comparan con los iniciales.
- **Dónde están los ficheros iniciales sin versionar:** los JSON por entrenamiento y los pesos
  de la ronda inicial se guardaron en `outputs/round3_initial_run/` (en el equipo de
  referencia; no están en git).

Los JSON de cada entrenamiento (unos 1000) quedan en `outputs/round3/` y los pesos en
`models/round3/`, sin versionar (ajuste 2 de D-024); se versiona un modelo representativo por red
en `models/round3/representative/`. El informe y las 15 gráficas de `docs/figures/round3/` se
regeneran desde estos tres ficheros con `report`.

## Rondas 1 y 2 (F3)

Resúmenes versionados de las dos rondas de F3 (D-020). Los resultados completos están en
`outputs/results/` (segunda ronda) y `outputs/results/eval1/` (primera), que no se versionan;
estos resúmenes salen de ellos con `tools/summarize_results.py`, sin entrenar nada ni volver a
puntuar test.

| Fichero | Qué contiene |
|---|---|
| [eval1.json](eval1.json) | Primera ronda: configuración y val de cada modelo, test (por segmento, por grabación y dentro de cada época), test de permutación, interpretabilidad, SHA-256 de los pesos. |
| [eval2.json](eval2.json) | Segunda ronda, igual, tras los arreglos de la revisión de F3 (F3-T12). |

## Por qué hay dos rondas

Test se evaluó por primera vez en F3-T09. La revisión de F3 encontró después fallos de
implementación y de medida (no de datos): posiciones muertas en la rama DEMON de la CNN, un
coseno de lr que no actuaba, el generador del aumento compartido con el del barajado, la Pd a
Pfa = 1 % interpolada con pocos negativos, un azar fijo de 0,25 y un test de permutación que no
repetía la elección de C (ajuste 2 de D-018). David aprobó arreglarlos **sin cambiar nada para
mejorar la cifra**, reentrenar las dos CNN con el mismo protocolo y evaluar test **una segunda y
última vez**. Las mejoras contra el sobreajuste quedan para F5.

Como test ya se había mirado, la segunda ronda no es una medida tan limpia como la primera: los
arreglos se decidieron por la revisión del código, no por las cifras de test, pero se hicieron
después de verlas.

## Comparación (F1 macro)

| Modelo | Ronda | Val (segmento) | Val, media de 3 semillas | Test (segmento) | Test (grabación) | Test 2016: Passeng. vs Tanker | Test 2017: Cargo vs Tug |
|---|---|---|---|---|---|---|---|
| Línea base | 1 | 0,381 | — | 0,374 | 0,444 | 0,527 | 0,470 |
| Línea base | 2 | 0,381 | — | 0,374 | 0,444 | 0,527 | 0,470 |
| CNN | 1 | 0,376 | 0,339 ± 0,028 | 0,167 | 0,111 | 0,276 | 0,656 |
| CNN | 2 | 0,391 | 0,329 ± 0,069 | 0,115 | 0,111 | 0,276 | 0,625 |
| CNN con aumento | 1 | 0,536 | 0,388 ± 0,110 | 0,185 | 0,290 | 0,463 | 0,421 |
| CNN con aumento | 2 | 0,358 | 0,297 ± 0,044 | 0,252 | 0,267 | 0,594 | 0,427 |
| *Azar medido (ronda 2)* | | | | *0,238* | *0,212* | *0,493* | *0,468* |

- La línea base es idéntica en las dos rondas (mismos pesos, SHA-256 `1968677a…`).
- Test de permutación de la línea base (validación cruzada por buque, 200 permutaciones entre
  buques): ronda 1, p = 0,040 con C fijo; ronda 2, p = 0,040 eligiendo C dentro de cada
  permutación (real 0,370 frente a media nula 0,272).
- En la ronda 1 el azar se tomaba como 0,25 (4 clases) y 0,5 (dentro de época).

## Cómo reproducir las rondas 1 y 2

Instrucciones que estaban en el README principal hasta `v0.1.1`. Necesitan los datos y los
rasgos (README principal, Puesta en marcha). Las rutas de los comandos son relativas a la
carpeta raíz del repositorio.

### Evaluar con los pesos de la ronda 2

Evalúa **solo los 3 modelos de las rondas 1 y 2** (línea base, CNN y CNN con aumento), con sus
pesos ya versionados y sin entrenar nada. Puntúa la partición fija de esas rondas con los pesos
de `models/` (en total menos de 1 MB). Con los rasgos ya
calculados (`features`, sección anterior):

```bash
docker compose run --rm pipeline evaluate
```

Tarda unos 30 segundos en CPU. Puntúa la partición de test con los tres modelos y escribe
`outputs/results/metrics.json` y las figuras `outputs/figures/results_*.png`. Debe mostrar estas
cifras de F1 macro por segmento, idénticas a las del informe (`docs/results/eval2.json`). La
reproducción exacta está comprobada en x86-64 (ver "Equipo de referencia" en el README principal); en otros
procesadores, sobre todo ARM (Apple Silicon), puede variar la última cifra:

| Modelo | Debe salir |
|---|---|
| `baseline` | `segment F1 0.374 (0.27-0.47)` |
| `cnn` | `segment F1 0.115 (0.07-0.16)` |
| `cnn_aug` | `segment F1 0.252 (0.17-0.34)` |

Para comprobarlo de una vez (mismos pesos y todas las cifras de test dentro de ±0,005, que
admite variaciones de la última cifra en otros procesadores):

```bash
docker compose run --rm --entrypoint python pipeline tools/check_reproduction.py
```

Debe decir `Reproduces the reference evaluation`.

Si comparas tu `metrics.json` con `docs/results/eval2.json`, el `code_sha256` será distinto: el
código cambió después de la evaluación 2 sin afectar a las cifras (detalle en la tabla de
procedencia de abajo). Las métricas y los `weights_sha256` deben
coincidir.

Otras opciones de `evaluate`:

- `--plot-only`: redibuja las figuras desde `metrics.json` sin volver a puntuar.
- `--interpretability`: qué bandas de frecuencia usa cada modelo (sobre val, unos 30 s).

### Reentrenar los modelos de las rondas 1 y 2

Son los tres modelos de las rondas 1 y 2. Reentrenar **sobrescribe los pesos de `models/`**. Para volver a los versionados:
`git checkout -- models/`.

| Comando | Qué hace | CPU | GPU (opción B) |
|---|---|---|---|
| `docker compose run --rm pipeline train --model duration` | Comprobación de DT-005: clasificador que solo ve la duración de la grabación | segundos | — |
| `docker compose run --rm pipeline train --model baseline` | Línea base (regresión logística) | ~30 s | (usa CPU) |
| `docker compose run --rm pipeline train --model cnn` | CNN: 4 configuraciones y 3 semillas | ~2-3 min (estimado) | ~40 s (23 s de entrenamiento) |
| `docker compose run --rm pipeline train --model cnn-aug` | CNN con aumento de datos (necesita `cnn` antes) | ~1-2 min (estimado) | ~30 s (15 s de entrenamiento) |
| `docker compose run --rm pipeline train --model baseline --permutation-test 200` | Test de permutación de la línea base | ~18 min | (usa CPU) |

- Los tiempos son los del equipo de referencia (abajo) e incluyen arrancar el contenedor; los de
  las CNN en CPU son una estimación a partir del tiempo por época medido.
- La línea base y la comprobación de duración dan exactamente los mismos resultados en x86-64;
  en otros procesadores puede variar la última cifra.
- Las CNN son deterministas en el mismo tipo de dispositivo, pero los pesos versionados se
  entrenaron en GPU (una RTX 4060): reentrenar en CPU, o en otra GPU, da pesos y cifras algo
  distintos.

## Procedencia: qué código produjo cada resultado

Regla (D-020): todo resultado que se conserva registra el código que lo produjo; si el código
cambia después, aquí se anotan las diferencias y si afectan a las cifras.

El `code_sha256` es el hash de los `.py` de `src/sonar_pipeline/` (`features.code_sha256`). Para
los commits de la tabla se recalculó desde el historial de git; los JSON de la ronda 2 lo
registran y coinciden con `ab383dd`. El **código de v0.1** (final de F4) tiene `code_sha256`
`b4b6638e…`. Lo que cambió en F4 (figura de datos, `report`, CLI, usuario del contenedor) no
toca el cálculo de rasgos, modelos ni métricas, así que no afecta a ninguna cifra.

| Resultado | Commit del código | `code_sha256` | Diferencias con el código de v0.1 (`b4b6638e…`) | ¿Afectan a las cifras? |
|---|---|---|---|---|
| Ronda 1 · línea base y su permutación | `6a08569` | `3e96833a…` | Permutación con C fijo (sin elegirlo dentro); sin hash de código ni hardware en los JSON | Sí en la permutación (p igual, 0,040, al repetirla) |
| Ronda 1 · CNN | `43cfb1e` | `06fdd962…` | Pooling con ceros (2 de 8 posiciones de DEMON siempre a 0); lr en coseno sin efecto; hilos sin fijar | **Sí** (por eso se reentrenó) |
| Ronda 1 · CNN con aumento | `ed9db91` | `b3abb7c3…` | Lo anterior, más el generador del aumento compartido con el barajado | **Sí** |
| Ronda 1 · evaluación de test | `b23c8ac` | `96168c20…` | Pd a Pfa = 1 % interpolada con < 100 negativos; azar fijo 0,25; hilos sin fijar | Sí en esas métricas; no en F1 ni en las matrices |
| Ronda 1 · interpretabilidad | `d544a71` | `19490ca8…` | Usó los modelos de la ronda 1 | Sí (otros modelos) |
| Ronda 2 · entrenamiento (las tres) | `ab383dd` | `3393ae4f…` | Sin el arreglo B5 (línea de azar en la figura de F1 si faltara una época) | No (solo el dibujo, y hoy no falta ninguna época) |
| Ronda 2 · evaluación de test | `ab383dd` | `3393ae4f…` | `evaluate` sin hilos fijados: corrió con 14 hilos (el entrenamiento, con 4); sin el campo `evaluation` (añadido después con `evaluate --refresh`, sin puntuar test); sin el arreglo B5 | No. **Comprobado en F4-T06** (fila siguiente): con el código final y 4 hilos, las cifras salen idénticas |
| Reproducción de la evaluación 2 (F4-T06, friend-tester, clon limpio) | `b581d0c` | `5abdb24b…` | Código final de F4 con 4 hilos; puntúa test solo para comprobar que se reproduce, sin ajustar nada | No: F1, intervalos y `weights_sha256` idénticos a `eval2.json` en los tres modelos |
| Ronda 2 · permutación e interpretabilidad | `ab383dd` | `3393ae4f…` | Ninguna que les afecte | No |
| Ronda 2 · comprobación de DT-005 (duración, solo val) | `5d8d820` | `aade7670…` | Regenerada para quitar el azar fijo y la Pd interpolada; después solo cambió lo de F4 | No |
| Figuras de la ronda 2 (`docs/figures/eval2/`) | `5d8d820` (`--plot-only`) | `aade7670…` | Redibujadas desde `metrics.json` con el arreglo B5, sin puntuar test | No |
| Figuras de la ronda 1 (`docs/figures/eval1/`) | `b23c8ac` (resultados) y `d544a71` (interpretabilidad) | `96168c20…` y `19490ca8…` | Copiadas tal cual: azar dibujado en 0,25 y 0,5; Pd a Pfa = 1 % interpolada con < 100 negativos; modelos de la ronda 1 | Sí en esas marcas; no en F1 ni en las matrices |
| Figuras de señal (`docs/figures/signal/`) | `9f1f99b` (`features --figures` y tests de figuras, F2-T10) | `c867b025…` | Mismo LOFAR y DEMON que el código final (después solo cambiaron los modelos y la evaluación) | No |
| Figura de datos e informe (`docs/figures/data_minutes.png`, `docs/report.md`) hasta `v0.1.1` | commit de F4-T07 (etiqueta `v0.1`) | `b4b6638e…` | Es el código de v0.1; `report` los regenera desde lo versionado con los mismos bytes (lo comprueba un test) | — |
| Ronda 3 inicial · entrenamiento, test y comparaciones (`round3_initial/round3.json`) | `c96865f` | `50fb6ba4…` | Calentamiento del lr que se quedaba en el 82 % (transformer e híbrido); sin norma del gradiente ni tiempos de carga | **Sí** en las 4 redes con calentamiento (por eso se reentrenó); no en la línea base ni la CNN |
| Ronda 3 inicial · interpretabilidad (`round3_initial/round3_interpretability.json`) | `8558273` | `b9420ed3…` | Usó los modelos iniciales | Sí (otros modelos) |
| Ronda 3 inicial · detalles (`round3_initial/round3_details.json`) | `1359477` | `cc71b4fe…` | Usó los modelos iniciales | Sí (otros modelos) |
| **Ronda 3 reentrenada** · entrenamiento, test, comparaciones, interpretabilidad y detalles (`round3.json`, `round3_interpretability.json`, `round3_details.json`, `models/round3/representative/`) | `7df7a96` (F5-T10) | `07c78f25…` | Es el código final de F5 | — |
| Ronda 3 · 15 figuras e informe (`docs/figures/round3/`, `docs/report.md`) | `839876b` (cierre de F5-T10; después, F5-T11) | `07c78f25…` | `report` los regenera desde lo versionado con los mismos bytes | — |

Los pesos de cada ronda están en git: ronda 1 en `6a08569` (línea base), `43cfb1e` (CNN) y
`ed9db91` (CNN con aumento); ronda 2 en `b43315c`.

## Hechos que el informe de F4 debe recoger

- Tug tiene 3 buques (una sola grabación en test): su resultado no es fiable, y las demás clases
  también tienen pocos buques.
- La CNN con aumento de la ronda 2 da en val, por grabación, un F1 macro de 0,071, por debajo del
  azar (0,199).
- El C elegido para la línea base (10) está en el borde de la rejilla.
- La validación cruzada mezcla folds de 3 y 4 clases (Tug falta en 3 de 5).
- La ROC es de clasificación, no de detección (DT-007).
