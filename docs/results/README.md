# Resultados de F3 · dos rondas de entrenamiento y evaluación

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

## Procedencia: qué código produjo cada resultado

Regla (D-020): todo resultado que se conserva registra el código que lo produjo; si el código
cambia después, aquí se anotan las diferencias y si afectan a las cifras.

El `code_sha256` es el hash de los `.py` de `src/sonar_pipeline/` (`features.code_sha256`). Para
los commits de la tabla se recalculó desde el historial de git; los JSON de la ronda 2 lo
registran y coinciden con `ab383dd`.

| Resultado | Commit del código | `code_sha256` | Diferencias con el código final (`5d8d820`) | ¿Afectan a las cifras? |
|---|---|---|---|---|
| Ronda 1 · línea base y su permutación | `6a08569` | `3e96833a…` | Permutación con C fijo (sin elegirlo dentro); sin hash de código ni hardware en los JSON | Sí en la permutación (p igual, 0,040, al repetirla) |
| Ronda 1 · CNN | `43cfb1e` | `06fdd962…` | Pooling con ceros (2 de 8 posiciones de DEMON siempre a 0); lr en coseno sin efecto; hilos sin fijar | **Sí** (por eso se reentrenó) |
| Ronda 1 · CNN con aumento | `ed9db91` | `b3abb7c3…` | Lo anterior, más el generador del aumento compartido con el barajado | **Sí** |
| Ronda 1 · evaluación de test | `b23c8ac` | `96168c20…` | Pd a Pfa = 1 % interpolada con < 100 negativos; azar fijo 0,25; hilos sin fijar | Sí en esas métricas; no en F1 ni en las matrices |
| Ronda 1 · interpretabilidad | `d544a71` | `19490ca8…` | Usó los modelos de la ronda 1 | Sí (otros modelos) |
| Ronda 2 · entrenamiento (las tres) | `ab383dd` | `3393ae4f…` | Sin el arreglo B5 (línea de azar en la figura de F1 si faltara una época) | No (solo el dibujo, y hoy no falta ninguna época) |
| Ronda 2 · evaluación de test | `ab383dd` | `3393ae4f…` | `evaluate` sin hilos fijados: corrió con 14 hilos (el entrenamiento, con 4); sin el campo `evaluation` (añadido después con `evaluate --refresh`, sin puntuar test); sin el arreglo B5 | En la práctica, no: el número de hilos solo podría cambiar el último decimal en CPU. **No se ha comprobado**, porque hacerlo exigiría mirar test una tercera vez |
| Ronda 2 · permutación e interpretabilidad | `ab383dd` | `3393ae4f…` | Ninguna que les afecte | No |
| Ronda 2 · comprobación de DT-005 (duración, solo val) | `5d8d820` | `aade7670…` | Es el código final; regenerada para quitar el azar fijo y la Pd interpolada | — |
| Figuras de la ronda 2 | `5d8d820` (`--plot-only`) | `aade7670…` | Redibujadas desde `metrics.json` con el arreglo B5, sin puntuar test | — |
| Figuras de señal (`docs/figures/signal/`) | `9f1f99b` (`features --figures` y tests de figuras, F2-T10) | `c867b025…` | Mismo LOFAR y DEMON que el código final (después solo cambiaron los modelos y la evaluación) | No |
| Figura de datos e informe (`docs/figures/data_minutes.png`, `docs/report.md`) | commit de F4-T03 (`report`) | — | Se regeneran con `report` desde lo versionado; la regeneración da los mismos bytes | — |

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
