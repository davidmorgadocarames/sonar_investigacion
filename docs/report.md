# Informe de resultados · sonar_investigacion

*Generado por `python -m sonar_pipeline report` a partir de `docs/results/` (D-020, D-021). No editar a mano.*

## Resumen

### Ronda 3 (F5): evaluación principal

- **Ronda 3 (evaluación principal, validación cruzada 5 × 5 por buque):** F1 macro de test por segmento entre 0,239 (Transformer) y 0,336 (Híbrido), con un azar medido de 0,242.
- **Ninguna de las 13 comparaciones fijadas de antemano es significativa** con la prueba de Nadeau y Bengio y la corrección de Holm. La más cercana: Transformer − Línea base, -0,092 (p 0,075, Holm 0,970).
- Las redes varían más entre particiones (desviación de 0,09 a 0,14) que la línea base (0,05): su resultado depende de qué buques caen en test.

### Rondas 1 y 2 (F3): partición fija, histórico

- En test (4 clases, por segmento) supera al azar medido (0,238) con todo su intervalo: **Línea base** 0,374 (0,27-0,47).
- CNN sin aumento 0,115 (0,07-0,16), por debajo del azar; CNN con aumento 0,252 (0,17-0,34), sin distinguirse del azar.
- Validación cruzada por buque sobre train + val (no es test): línea base 0,370 ± 0,113; test de permutación entre buques, **p = 0,040**.
- Dentro de cada época de grabación (Cargo frente a Tug, Passengership frente a Tanker) ningún modelo supera al azar con todo su intervalo: con estos datos **no se puede afirmar que se reconozca la firma del buque** (ver Limitaciones).

## 1. Datos

Subconjunto público de DeepShip (63 grabaciones, 43 buques, 95,4 minutos de audio; 4 clases). Toda partición es **por buque**: ningún buque aparece en dos papeles a la vez (D-007, D-013, D-024).

| Clase | Minutos | Buques | Grabaciones |
|---|---|---|---|
| Cargo | 38,7 | 11 | 12 |
| Passengership | 19,1 | 10 | 20 |
| Tanker | 27,7 | 19 | 28 |
| Tug | 9,9 | 3 | 3 |

![Datos por clase](figures/data_classes.png)

![Las 25 particiones](figures/data_partitions.png)

## 2. Señal

Cada grabación se centra (offset DC) y se corta en segmentos de 3 s sin solapamiento, escalados a RMS = 1 (D-015).

- **LOFAR:** diezmado a 5 kHz, ventana Hann de 0,50 s con avance de 0,25 s, 0-2000 Hz (11 tramas × 1001 bins de 2 Hz), normalizado con TPSW (dB sobre el fondo).
- **DEMON:** paso banda 2000-8000 Hz, envolvente de Hilbert diezmada a 500 Hz, espectro de 0 a 100 Hz (301 bins de 1/3 Hz).

Validación con señales sintéticas: un tono de 440 Hz aparece en su bin del LOFAR y una modulación de 12 Hz en el del DEMON.

![Validación LOFAR](figures/signal/validation_lofar.png)
![Validación DEMON](figures/signal/validation_demon.png)

Ejemplos (la grabación de entrenamiento más larga de cada clase):

![Ejemplo Cargo](figures/signal/example_cargo.png)
![Ejemplo Passengership](figures/signal/example_passengership.png)
![Ejemplo Tanker](figures/signal/example_tanker.png)
![Ejemplo Tug](figures/signal/example_tug.png)

## 3. Modelos y protocolo

- **Línea base:** regresión logística estandarizada con pesos por clase; C elegido con val; validación cruzada agrupada por buque y test de permutación entre buques.
- **CNN:** dos ramas (LOFARgram 2D y DEMON 1D), unos 79 000 parámetros, AdamW, pesos por clase y parada temprana con el F1 macro de val; rejilla 2 × 2 y 3 semillas.
- **CNN con aumento:** la misma red, configuración y semillas, con enmascarado de frecuencia y tiempo y pequeños desplazamientos solo en entrenamiento.
- Todo se decide con val; test se usa solo para la evaluación final de cada ronda. La decisión por grabación es la media de las probabilidades de sus segmentos (D-018).
- **Ronda 3 (F5):** además, un **transformer** (bandas de 40 Hz del LOFAR, un token del DEMON y uno de clase; RMSNorm, RoPE, GQA y SwiGLU) y un **híbrido** (dos bloques de la CNN como tokens y bloques transformer), cada red sin y con aumento, con 4 configuraciones cada una y validación cruzada 5 × 5 por buque (D-024). Arquitecturas y justificación de cada configuración en el README.

## 4. Ronda 3 (F5): validación cruzada 5 × 5 · evaluación principal

**Resultados reentrenados (F5-T10).** Los resultados iniciales de la ronda 3 tenían un fallo del calentamiento del lr en el transformer y el híbrido, sin y con aumento: al terminar las 5 épocas de calentamiento el lr se quedaba en el 82 % (4,1e-4 en lugar de 5e-4; ajuste 3 de D-024). Se arregló y se reentrenó la ronda completa **después de haber visto test**: igual que la ronda 2, es un arreglo de implementación decidido por la revisión del código, no por las cifras. La CNN y la línea base no tenían calentamiento. Los resultados iniciales se conservan en `docs/results/round3_initial/`.

| Modelo | Config. inicial | Config. reentrenada | Test inicial | Test reentrenado | Diferencia |
|---|---|---|---|---|---|
| Línea base | E | E | 0,331 ± 0,046 | 0,331 ± 0,046 | 0,000 |
| CNN | C | C | 0,316 ± 0,135 | 0,316 ± 0,135 | 0,000 |
| CNN + aum. | D | D | 0,299 ± 0,130 | 0,299 ± 0,130 | 0,000 |
| Híbrido | B | B | 0,343 ± 0,126 | 0,336 ± 0,126 | -0,007 |
| Híbrido + aum. | A | A | 0,340 ± 0,139 | 0,304 ± 0,136 | -0,036 |
| Transformer | B | B | 0,235 ± 0,080 | 0,239 ± 0,085 | 0,004 |
| Transformer + aum. | B | B | 0,240 ± 0,085 | 0,249 ± 0,092 | 0,008 |

Protocolo fijado antes de entrenar (D-024 y sus ajustes): 25 particiones 60/20/20 por buque (5 repeticiones × 5 folds; Tug, con 3 buques, 1/1/1 en todas); la configuración de cada modelo se elige con la media de validación de las 25 particiones y la semilla 0, sin mirar test; el modelo final promedia las probabilidades de las semillas 0, 1, 2. Ningún modelo está preentrenado. Comparaciones con la prueba t corregida de Nadeau y Bengio (relación test/entrenamiento 0,375) y corrección de Holm.

F1 macro de test por segmento (media ± desviación de las 25 particiones):

| Modelo | Config. | Val (elección) | **Test, 4 clases** | Mín.-máx. | Test por grabación | 2016: Pasaje vs Tanker | 2017: Cargo vs Tug | ECE | MFLOPs | Parámetros |
|---|---|---|---|---|---|---|---|---|---|---|
| Línea base | E | 0,337 | **0,331 ± 0,046** | 0,21-0,41 | 0,392 ± 0,104 | 0,527 ± 0,093 | 0,527 ± 0,146 | 0,45 | 0,01 | 5.212 |
| CNN | C | 0,399 | **0,316 ± 0,135** | 0,15-0,56 | 0,291 ± 0,181 | 0,562 ± 0,123 | 0,660 ± 0,231 | 0,20 | 75,33 | 79.076 |
| CNN + aum. | D | 0,398 | **0,299 ± 0,130** | 0,10-0,51 | 0,293 ± 0,145 | 0,586 ± 0,115 | 0,627 ± 0,209 | 0,20 | 75,33 | 79.076 |
| Híbrido | B | 0,410 | **0,336 ± 0,126** | 0,13-0,57 | 0,310 ± 0,161 | 0,506 ± 0,092 | 0,727 ± 0,201 | 0,27 | 57,32 | 76.132 |
| Híbrido + aum. | A | 0,408 | **0,304 ± 0,136** | 0,11-0,55 | 0,243 ± 0,171 | 0,485 ± 0,132 | 0,645 ± 0,194 | 0,24 | 52,23 | 32.036 |
| Transformer | B | 0,322 | **0,239 ± 0,085** | 0,12-0,53 | 0,208 ± 0,107 | 0,526 ± 0,085 | 0,499 ± 0,133 | 0,27 | 6,93 | 80.068 |
| Transformer + aum. | B | 0,328 | **0,249 ± 0,092** | 0,14-0,53 | 0,211 ± 0,132 | 0,519 ± 0,091 | 0,539 ± 0,163 | 0,26 | 6,93 | 80.068 |
| *Azar medido* | | | *0,242* | | | *≈ 0,5* | *≈ 0,5* | | | |

Comparaciones fijadas de antemano:

| Comparación | Diferencia media | IC 95 % corregido | Gana en | p | p (Holm) |
|---|---|---|---|---|---|
| CNN − Línea base | -0,015 | -0,197 a 0,167 | 8/25 | 0,865 | 1,000 |
| CNN + aum. − Línea base | -0,032 | -0,205 a 0,141 | 9/25 | 0,708 | 1,000 |
| Híbrido − Línea base | 0,005 | -0,165 a 0,176 | 10/25 | 0,949 | 1,000 |
| Híbrido + aum. − Línea base | -0,027 | -0,203 a 0,148 | 11/25 | 0,752 | 1,000 |
| Transformer − Línea base | -0,092 | -0,194 a 0,010 | 4/25 | 0,075 | 0,970 |
| Transformer + aum. − Línea base | -0,082 | -0,188 a 0,024 | 4/25 | 0,122 | 1,000 |
| CNN + aum. − CNN | -0,017 | -0,085 a 0,052 | 8/25 | 0,622 | 1,000 |
| Híbrido + aum. − Híbrido | -0,032 | -0,179 a 0,114 | 9/25 | 0,651 | 1,000 |
| Transformer + aum. − Transformer | 0,010 | -0,039 a 0,059 | 16/25 | 0,677 | 1,000 |
| Transformer − CNN | -0,077 | -0,239 a 0,085 | 8/25 | 0,338 | 1,000 |
| Híbrido − CNN | 0,020 | -0,112 a 0,153 | 17/25 | 0,753 | 1,000 |
| Transformer + aum. − CNN + aum. | -0,050 | -0,202 a 0,101 | 9/25 | 0,498 | 1,000 |
| Híbrido + aum. − CNN + aum. | 0,005 | -0,125 a 0,135 | 13/25 | 0,942 | 1,000 |

Configuraciones (F1 medio de validación; en negrita, la elegida):

- Línea base: A 0,312, B 0,315, C 0,316, D 0,318, **E 0,337**.
- CNN: A 0,396, B 0,369, **C 0,399**, D 0,380.
- CNN + aum.: A 0,397, B 0,380, C 0,390, **D 0,398**.
- Híbrido: A 0,403, **B 0,410**, C 0,387, D 0,339.
- Híbrido + aum.: **A 0,408**, B 0,393, C 0,393, D 0,379.
- Transformer: A 0,289, **B 0,322**, C 0,311, D 0,313.
- Transformer + aum.: A 0,308, **B 0,328**, C 0,317, D 0,314.

Interpretabilidad (sobre validación, modelos finales):

- Línea base: bandas principales 1240-1280, 1280-1320, 0-40, 1320-1360, 1440-1480 Hz.
- CNN: bandas principales 0-40, 1240-1280, 720-760, 480-520, 600-640 Hz.
- CNN + aum.: bandas principales 0-40, 1240-1280, 160-200, 200-240, 120-160 Hz.
- Híbrido: bandas principales 0-40, 40-80, 400-440, 360-400, 80-120 Hz; atención del token de clase al DEMON 13 %.
- Híbrido + aum.: bandas principales 0-40, 40-80, 120-160, 160-200, 80-120 Hz; atención del token de clase al DEMON 20 %.
- Transformer: bandas principales 0-40, 240-280, 40-80, 440-480, 320-360 Hz; atención del token de clase al DEMON 11 %.
- Transformer + aum.: bandas principales 0-40, 40-80, 240-280, 320-360, 840-880 Hz; atención del token de clase al DEMON 15 %.
- Bandas que más difieren entre épocas: 40, 1240, 1320, 1400, 1440 Hz; líneas fijas de época: ninguna.
- Coincidencia de las 5 bandas principales con esas 5: Línea base 3/5; CNN 1/5; CNN + aum. 1/5; Híbrido 1/5; Híbrido (atención) 1/5; Híbrido + aum. 1/5; Híbrido + aum. (atención) 1/5; Transformer 1/5; Transformer (atención) 1/5; Transformer + aum. 1/5; Transformer + aum. (atención) 1/5. Las correlaciones con la diferencia entre épocas no son fiables (bandas vecinas correlacionadas, muchos modelos sin corrección).

### F1 de test de cada partición

![F1 de test de cada partición](figures/round3/round3_box.png)

Cada punto es una de las 25 particiones; la caja va del primer al tercer cuartil, la raya es la mediana y el rombo la media. La línea discontinua es el azar (medido con 4 clases; 0,5 aproximado dentro de cada época, donde solo hay 2 clases). Lo importante es lo **alta** que es cada caja: dice cuánto cambia el resultado según qué buques le tocan al modelo en test.

### Comparaciones fijadas de antemano

![Comparaciones fijadas de antemano](figures/round3/round3_pairs.png)

Diferencia media de F1 entre dos modelos en las mismas 25 particiones, con su intervalo del 95 % corregido por Nadeau y Bengio (que tiene en cuenta que las particiones comparten buques). Si el intervalo cruza el 0, la diferencia no se distingue del ruido. A la derecha, el p sin corregir y con Holm (13 comparaciones a la vez).

### Efecto del aumento de datos

![Efecto del aumento de datos](figures/round3/round3_augmentation.png)

Para cada partición, F1 con aumento menos F1 sin aumento de la misma arquitectura. Por encima de 0, el aumento ayudó en esa partición.

### Precisión frente a coste

![Precisión frente a coste](figures/round3/round3_cost.png)

F1 medio (± desviación) frente al coste de inferencia, al tamaño y al tiempo de entrenamiento, en escala logarítmica. Un modelo compensa si está más arriba sin irse mucho a la derecha.

### Velocidad y memoria

![Velocidad y memoria](figures/round3/round3_speed.png)

Medidas en la RTX 4060 (la línea base, en CPU). Los tokens por segundo solo existen en el transformer y el híbrido. 'Formar minibatches' es la parte del tiempo de entrenamiento que se va en preparar cada lote: casi toda es el aumento de datos, así que solo pesa en las versiones con aumento.

### Curvas de aprendizaje

![Curvas de aprendizaje](figures/round3/round3_curves.png)

Media (y banda de ± 1 desviación) de los entrenamientos finales, época a época. Si la pérdida de entrenamiento baja mientras la de validación sube, el modelo está memorizando: es el sobreajuste. El último panel es la norma del gradiente (media por época, en escala logarítmica): si crece, los pasos del entrenamiento se hacen más grandes y menos estables.

### Brecha de generalización

![Brecha de generalización](figures/round3/round3_generalisation.png)

F1 del modelo final (media de sus semillas) en entrenamiento, validación y test. La distancia entre el círculo y el rombo es lo que el modelo memoriza y no generaliza.

### De dónde viene la variación

![De dónde viene la variación](figures/round3/round3_variance.png)

Desviación del F1 de test entre particiones (con la media de semillas) y entre semillas dentro de una misma partición. Es el análisis de Bouthillier et al. (2021): justifica usar muchas particiones antes que muchas semillas.

### Elección de configuración

![Elección de configuración](figures/round3/round3_selection.png)

F1 de validación (media ± desviación de las 25 particiones) de cada configuración; la rellena es la elegida y el rombo su F1 de test. Si las barras se solapan mucho, la elección es casi un empate.

### Matrices de confusión conjuntas

![Matrices de confusión conjuntas](figures/round3/round3_confusion.png)

Todas las predicciones de test de las 25 particiones juntas (cada segmento cuenta 5 veces, una por repetición). Filas: clase real; columnas: clase predicha.

### Acierto por buque

![Acierto por buque](figures/round3/round3_vessels.png)

Fracción de las veces que cada buque estuvo en test en que sus grabaciones se clasificaron bien. Muestra los buques que todos los modelos fallan.

### Acuerdo entre modelos

![Acuerdo entre modelos](figures/round3/round3_agreement.png)

Para cada pareja de modelos, en qué fracción de las grabaciones de test predicen lo mismo y en qué fracción fallan los dos. Si dos modelos fallan en sitios distintos, combinarlos podría ayudar.

### Calibración

![Calibración](figures/round3/round3_calibration.png)

Si el modelo dice "80 % seguro", ¿acierta el 80 % de las veces? La diagonal es la calibración perfecta; por debajo, el modelo está demasiado seguro. ECE y Brier: cuanto más bajos, mejor.

### Qué frecuencias usa cada modelo

![Qué frecuencias usa cada modelo](figures/round3/round3_interpretability.png)

Arriba, cuánto depende cada modelo de cada banda de 40 Hz (normalizado por fila), y en la última fila cuánto difiere esa banda entre las grabaciones de 2016 y 2017. Si un modelo usa las bandas que delatan la época, puede estar aprendiendo la época y no el buque. Abajo, a qué bandas atiende el token de clase del transformer y del híbrido.

### ROC de clasificación

![ROC de clasificación](figures/round3/round3_roc.png)

Una clase frente al resto, con los segmentos de test de las 25 particiones juntos. Es una ROC de clasificación, no de detección (DT-007). Al juntar particiones se mezclan 25 modelos distintos, así que el AUC es orientativo.

## 5. Rondas 1 y 2 (F3): partición fija · histórico

**Etapa anterior, no comparable directamente con la ronda 3:** una sola partición fija (12 buques en test), mirada dos veces. F1 macro con intervalos del 95 % (bootstrap remuestreando grabaciones dentro de cada clase). Hubo **dos rondas**: la segunda tras arreglar los fallos de implementación y de medida que encontró la revisión de F3, sin cambiar nada para mejorar la cifra; test se había mirado ya en la primera. Detalle y procedencia en [docs/results/README.md](results/README.md).

### Ronda 2 (final)

Modelos versionados en `models/` (los que usa `evaluate`).

| Modelo | Val (segmento) | Val, media de semillas | Test (segmento) | Test (grabación) | Test 2016: Passeng. vs Tanker | Test 2017: Cargo vs Tug |
|---|---|---|---|---|---|---|
| Línea base | 0,381 | — | 0,374 (0,27-0,47) | 0,444 (0,23-0,61) | 0,527 (0,42-0,70) | 0,470 (0,35-0,59) |
| CNN sin aumento | 0,391 | 0,329 ± 0,069 | 0,115 (0,07-0,16) | 0,111 (0,11-0,11) | 0,276 (0,15-0,36) | 0,625 (0,39-0,77) |
| CNN con aumento | 0,358 | 0,297 ± 0,044 | 0,252 (0,17-0,34) | 0,267 (0,09-0,37) | 0,594 (0,35-0,84) | 0,427 (0,43-0,43) |
| *Azar medido* | | | *0,238* | *0,212* | *0,493* | *0,468* |

Test de permutación de la línea base (200 permutaciones de las clases entre buques; eligiendo C con val dentro de cada permutación): F1 real 0,370 frente a una media nula de 0,272 (percentil 95: 0,364); **p = 0,040**.

Interpretabilidad (sobre val):

- Línea base, bandas de LOFAR más importantes: 40-80 Hz, 1240-1280 Hz, 1440-1480 Hz, 400-440 Hz, 200-240 Hz.
- CNN sin aumento, oclusión: 0-40 Hz, 80-120 Hz, 40-80 Hz, 160-200 Hz, 1320-1360 Hz.
- CNN con aumento, oclusión: 0-40 Hz, 80-120 Hz, 440-480 Hz, 960-1000 Hz, 920-960 Hz.
- Líneas fijas de época: ninguna.
- Relación con las bandas que más difieren entre épocas: Línea base ρ = 0,29 (p = 0,039), 2/5 bandas en común; CNN sin aumento ρ = 0,27 (p = 0,055), 2/5 bandas en común; CNN con aumento ρ = 0,02 (p = 0,881), 1/5 bandas en común. Ninguno de estos p es fiable (bandas vecinas correlacionadas, sin corrección).

![F1 con intervalos](figures/eval2/results_f1.png)
![Matrices de confusión](figures/eval2/results_confusion.png)
![ROC por segmento](figures/eval2/results_roc_segment.png)
![ROC por grabación](figures/eval2/results_roc_recording.png)
![Interpretabilidad por bandas](figures/eval2/interpretability_bands.png)

### Ronda 1

Primera evaluación de test, antes de los arreglos (figuras copiadas tal cual; el azar se suponía 0,25 y 0,5, y la Pd a Pfa = 1 % de sus ROC está interpolada con menos de 100 negativos).

| Modelo | Val (segmento) | Val, media de semillas | Test (segmento) | Test (grabación) | Test 2016: Passeng. vs Tanker | Test 2017: Cargo vs Tug |
|---|---|---|---|---|---|---|
| Línea base | 0,381 | — | 0,374 (0,27-0,47) | 0,444 (0,23-0,61) | 0,527 (0,42-0,70) | 0,470 (0,35-0,59) |
| CNN sin aumento | 0,376 | 0,339 ± 0,028 | 0,167 (0,09-0,23) | 0,111 (0,11-0,11) | 0,276 (0,15-0,36) | 0,656 (0,42-0,81) |
| CNN con aumento | 0,536 | 0,388 ± 0,110 | 0,185 (0,10-0,28) | 0,290 (0,16-0,41) | 0,463 (0,24-0,73) | 0,421 (0,41-0,43) |
| *Azar (supuesto)* | | | *0,25* | *0,25* | *0,5* | *0,5* |

Test de permutación de la línea base (200 permutaciones de las clases entre buques; C fijo): F1 real 0,370 frente a una media nula de 0,270 (percentil 95: 0,355); **p = 0,040**.

Interpretabilidad (sobre val):

- Línea base, bandas de LOFAR más importantes: 40-80 Hz, 1240-1280 Hz, 1440-1480 Hz, 400-440 Hz, 200-240 Hz.
- CNN sin aumento, oclusión: 0-40 Hz, 80-120 Hz, 1440-1480 Hz, 1360-1400 Hz, 1320-1360 Hz.
- CNN con aumento, oclusión: 0-40 Hz, 80-120 Hz, 1440-1480 Hz, 1240-1280 Hz, 440-480 Hz.
- Líneas fijas de época: ninguna.
- Relación con las bandas que más difieren entre épocas: Línea base ρ = 0,29 (p = 0,039), 2/5 bandas en común; CNN sin aumento ρ = 0,08 (p = 0,572), 3/5 bandas en común; CNN con aumento ρ = -0,05 (p = 0,743), 3/5 bandas en común. Ninguno de estos p es fiable (bandas vecinas correlacionadas, sin corrección).

![F1 con intervalos](figures/eval1/results_f1.png)
![Matrices de confusión](figures/eval1/results_confusion.png)
![ROC por segmento](figures/eval1/results_roc_segment.png)
![ROC por grabación](figures/eval1/results_roc_recording.png)
![Interpretabilidad por bandas](figures/eval1/interpretability_bands.png)

## 6. Limitaciones

- **Clase y época coinciden** (DT-005): Cargo y Tug se grabaron en 2017 y Passengership y Tanker en 2016. El resultado de 4 clases no distingue el tipo de buque de las condiciones de grabación; solo las comparaciones dentro de una época están libres de ese efecto, y ahí ningún modelo supera al azar con todo su intervalo. Un clasificador que solo ve la duración de la grabación obtiene en val un F1 macro de 0,932 por segmento y 0,800 por grabación (azar medido 0,246 y 0,199).
- **Pocos buques:** 43 en total. Tug tiene 3 (una sola grabación en test): su resultado no es fiable, y las demás clases también tienen pocos buques.
- **Modelos inestables:** la CNN con aumento guardada da en val, por grabación, 0,071 frente a un azar de 0,199; F1 de val entre semillas: CNN sin aumento 0,329 ± 0,069, CNN con aumento 0,297 ± 0,044.
- **La ROC es de clasificación** (una clase frente al resto), no de detección: DeepShip no tiene grabaciones de solo fondo (DT-007). La Pd a Pfa = 1 % no se puede medir con menos de 100 negativos.
- **Azar medido:** con clases desequilibradas el azar no es 0,25 exacto; se mide con un clasificador aleatorio sobre las etiquetas reales.
- **Dos evaluaciones de test:** la segunda se hizo después de ver la primera; los arreglos vinieron de la revisión del código, no de las cifras.
- La validación cruzada de la línea base mezcla folds de 3 y 4 clases (Tug falta en 3 de 5), y el C elegido (10) está en el borde de la rejilla.
- **Reproducción exacta comprobada en x86-64** (Intel Core i7-14700KF, imagen CPU); en otros procesadores, sobre todo ARM, puede variar la última cifra.
- **Ronda 3, Tug:** con 3 buques, cada uno cae en test 8 o 9 veces y siempre con los otros dos en entrenamiento y validación; el solapamiento entre particiones es mayor en esa clase que el que corrige la prueba de Nadeau y Bengio (ajuste 1 de D-024).
- **Ronda 3, azar dentro de época:** se marca en 0,5 aproximado; el azar medido de cada pareja no se guardó por partición.
- **Ronda 3, figuras conjuntas** (matrices, ROC, calibración): juntan predicciones de 25 modelos distintos; las cifras por partición son las de referencia.

## 7. Procedencia y reproducibilidad

- Qué código produjo cada resultado: tabla de procedencia en [docs/results/README.md](results/README.md).
- SHA-256 de los pesos evaluados (ronda 2): baseline `1968677ae9e5…`, cnn `bcda0f3385fd…`, cnn_aug `701214ab0c5c…`.
- Las figuras de señal se generaron con `features --figures` y los tests de figuras; las de resultados, con `evaluate` (la ronda 2 redibujada con `evaluate --plot-only`).

