# Informe de resultados · sonar_investigacion

*Generado por `python -m sonar_pipeline report` a partir de `docs/results/` (D-020, D-021). No editar a mano.*

## Resumen

### Ronda 3 (F5): evaluación principal

- **Ronda 3 (evaluación principal, validación cruzada 5 × 5 por buque):** F1 macro de test por segmento entre 0,235 (Transformer) y 0,343 (Híbrido), con un azar medido de 0,242.
- **Ninguna de las 13 comparaciones fijadas de antemano es significativa** con la prueba de Nadeau y Bengio y la corrección de Holm. La más cercana: Transformer − Línea base, -0,096 (p 0,052, Holm 0,680).
- Las redes varían más entre particiones (desviación de 0,08 a 0,14) que la línea base (0,05): su resultado depende de qué buques caen en test.

### Rondas 1 y 2 (F3): partición fija, histórico

- En test (4 clases, por segmento) supera al azar medido (0,238) con todo su intervalo: **Línea base** 0,374 (0,27-0,47).
- CNN sin aumento 0,115 (0,07-0,16), por debajo del azar; CNN con aumento 0,252 (0,17-0,34), sin distinguirse del azar.
- Validación cruzada por buque sobre train + val (no es test): línea base 0,370 ± 0,113; test de permutación entre buques, **p = 0,040**.
- Dentro de cada época de grabación (Cargo frente a Tug, Passengership frente a Tanker) ningún modelo supera al azar con todo su intervalo: con estos datos **no se puede afirmar que se reconozca la firma del buque** (ver Limitaciones).

## 1. Datos

Subconjunto público de DeepShip (63 grabaciones, 43 buques, 95,4 minutos de audio; 4 clases). La partición es **por buque**: ningún buque aparece en dos particiones (D-007, D-013).

| Clase | Entrenamiento | Validación | Test |
|---|---|---|---|
| Cargo | 22,9 min · 6 b · 7 g | 6,5 min · 2 b · 2 g | 9,3 min · 3 b · 3 g |
| Passengership | 11,4 min · 6 b · 12 g | 3,9 min · 2 b · 4 g | 3,8 min · 2 b · 4 g |
| Tanker | 16,6 min · 8 b · 16 g | 5,5 min · 5 b · 6 g | 5,5 min · 6 b · 6 g |
| Tug | 3,4 min · 1 b · 1 g | 3,3 min · 1 b · 1 g | 3,2 min · 1 b · 1 g |

b = buques, g = grabaciones.

![Minutos por clase y partición](figures/data_minutes.png)

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

Protocolo fijado antes de entrenar (D-024 y sus ajustes): 25 particiones 60/20/20 por buque (5 repeticiones × 5 folds; Tug, con 3 buques, 1/1/1 en todas); la configuración de cada modelo se elige con la media de validación de las 25 particiones y la semilla 0, sin mirar test; el modelo final promedia las probabilidades de las semillas 0, 1, 2. Ningún modelo está preentrenado. Comparaciones con la prueba t corregida de Nadeau y Bengio (relación test/entrenamiento 0,375) y corrección de Holm.

F1 macro de test por segmento (media ± desviación de las 25 particiones):

| Modelo | Config. | Val (elección) | **Test, 4 clases** | Mín.-máx. | Test por grabación | 2016: Pasaje vs Tanker | 2017: Cargo vs Tug | ECE | MFLOPs | Parámetros |
|---|---|---|---|---|---|---|---|---|---|---|
| Línea base | E | 0,337 | **0,331 ± 0,046** | 0,21-0,41 | 0,392 ± 0,104 | 0,527 ± 0,093 | 0,527 ± 0,146 | 0,45 | 0,01 | 5.212 |
| CNN | C | 0,399 | **0,316 ± 0,135** | 0,15-0,56 | 0,291 ± 0,181 | 0,562 ± 0,123 | 0,660 ± 0,231 | 0,20 | 75,33 | 79.076 |
| CNN + aum. | D | 0,398 | **0,299 ± 0,130** | 0,10-0,51 | 0,293 ± 0,145 | 0,586 ± 0,115 | 0,627 ± 0,209 | 0,20 | 75,33 | 79.076 |
| Híbrido | B | 0,406 | **0,343 ± 0,126** | 0,15-0,58 | 0,332 ± 0,179 | 0,513 ± 0,112 | 0,712 ± 0,244 | 0,26 | 57,32 | 76.132 |
| Híbrido + aum. | A | 0,403 | **0,340 ± 0,139** | 0,11-0,56 | 0,323 ± 0,178 | 0,494 ± 0,131 | 0,683 ± 0,208 | 0,20 | 52,23 | 32.036 |
| Transformer | B | 0,324 | **0,235 ± 0,080** | 0,12-0,50 | 0,210 ± 0,113 | 0,521 ± 0,082 | 0,509 ± 0,133 | 0,27 | 6,93 | 80.068 |
| Transformer + aum. | B | 0,325 | **0,240 ± 0,085** | 0,13-0,50 | 0,201 ± 0,109 | 0,529 ± 0,094 | 0,531 ± 0,149 | 0,27 | 6,93 | 80.068 |
| *Azar medido* | | | *0,242* | | | *≈ 0,5* | *≈ 0,5* | | | |

Comparaciones fijadas de antemano:

| Comparación | Diferencia media | IC 95 % corregido | Gana en | p | p (Holm) |
|---|---|---|---|---|---|
| CNN − Línea base | -0,015 | -0,197 a 0,167 | 8/25 | 0,865 | 1,000 |
| CNN + aum. − Línea base | -0,032 | -0,205 a 0,141 | 9/25 | 0,708 | 1,000 |
| Híbrido − Línea base | 0,013 | -0,159 a 0,184 | 12/25 | 0,880 | 1,000 |
| Híbrido + aum. − Línea base | 0,009 | -0,179 a 0,197 | 11/25 | 0,923 | 1,000 |
| Transformer − Línea base | -0,096 | -0,193 a 0,001 | 3/25 | 0,052 | 0,680 |
| Transformer + aum. − Línea base | -0,090 | -0,191 a 0,010 | 4/25 | 0,075 | 0,906 |
| CNN + aum. − CNN | -0,017 | -0,085 a 0,052 | 8/25 | 0,622 | 1,000 |
| Híbrido + aum. − Híbrido | -0,004 | -0,122 a 0,115 | 12/25 | 0,948 | 1,000 |
| Transformer + aum. − Transformer | 0,006 | -0,022 a 0,033 | 15/25 | 0,682 | 1,000 |
| Transformer − CNN | -0,081 | -0,239 a 0,077 | 7/25 | 0,301 | 1,000 |
| Híbrido − CNN | 0,028 | -0,108 a 0,163 | 17/25 | 0,676 | 1,000 |
| Transformer + aum. − CNN + aum. | -0,059 | -0,212 a 0,094 | 9/25 | 0,436 | 1,000 |
| Híbrido + aum. − CNN + aum. | 0,041 | -0,049 a 0,130 | 18/25 | 0,359 | 1,000 |

Configuraciones (F1 medio de validación; en negrita, la elegida):

- Línea base: A 0,312, B 0,315, C 0,316, D 0,318, **E 0,337**.
- CNN: A 0,396, B 0,369, **C 0,399**, D 0,380.
- CNN + aum.: A 0,397, B 0,380, C 0,390, **D 0,398**.
- Híbrido: A 0,400, **B 0,406**, C 0,385, D 0,357.
- Híbrido + aum.: **A 0,403**, B 0,376, C 0,387, D 0,376.
- Transformer: A 0,284, **B 0,324**, C 0,310, D 0,304.
- Transformer + aum.: A 0,302, **B 0,325**, C 0,319, D 0,314.

Interpretabilidad (sobre validación, modelos finales):

- Línea base: bandas principales 1240-1280, 1280-1320, 0-40, 1320-1360, 1440-1480 Hz.
- CNN: bandas principales 0-40, 1240-1280, 720-760, 480-520, 600-640 Hz.
- CNN + aum.: bandas principales 0-40, 1240-1280, 160-200, 200-240, 120-160 Hz.
- Híbrido: bandas principales 0-40, 40-80, 400-440, 80-120, 360-400 Hz; atención del token de clase al DEMON 14 %.
- Híbrido + aum.: bandas principales 0-40, 40-80, 160-200, 120-160, 80-120 Hz; atención del token de clase al DEMON 19 %.
- Transformer: bandas principales 0-40, 40-80, 440-480, 240-280, 840-880 Hz; atención del token de clase al DEMON 11 %.
- Transformer + aum.: bandas principales 0-40, 40-80, 240-280, 840-880, 440-480 Hz; atención del token de clase al DEMON 14 %.
- Bandas que más difieren entre épocas: 40, 1240, 1320, 1400, 1440 Hz; líneas fijas de época: ninguna.
- Coincidencia de las 5 bandas principales con esas 5: Línea base 3/5; CNN 1/5; CNN + aum. 1/5; Híbrido 1/5; Híbrido (atención) 1/5; Híbrido + aum. 1/5; Híbrido + aum. (atención) 1/5; Transformer 1/5; Transformer (atención) 1/5; Transformer + aum. 1/5; Transformer + aum. (atención) 1/5. Las correlaciones con la diferencia entre épocas no son fiables (bandas vecinas correlacionadas, muchos modelos sin corrección).

### F1 de test de cada partición

![Ronda 3 · F1 de test de cada partición](figures/round3/round3_box.png)

Cada punto es una de las 25 particiones; la caja va del primer al tercer cuartil, la raya es la mediana y el rombo la media. La línea discontinua es el azar (medido con 4 clases; 0,5 aproximado dentro de cada época, donde solo hay 2 clases). Lo importante es lo **alta** que es cada caja: dice cuánto cambia el resultado según qué buques le tocan al modelo en test.

### Comparaciones fijadas de antemano

![Ronda 3 · Comparaciones fijadas de antemano](figures/round3/round3_pairs.png)

Diferencia media de F1 entre dos modelos en las mismas 25 particiones, con su intervalo del 95 % corregido por Nadeau y Bengio (que tiene en cuenta que las particiones comparten buques). Si el intervalo cruza el 0, la diferencia no se distingue del ruido. A la derecha, el p sin corregir y con Holm (13 comparaciones a la vez).

### Efecto del aumento de datos

![Ronda 3 · Efecto del aumento de datos](figures/round3/round3_augmentation.png)

Para cada partición, F1 con aumento menos F1 sin aumento de la misma arquitectura. Por encima de 0, el aumento ayudó en esa partición.

### Precisión frente a coste

![Ronda 3 · Precisión frente a coste](figures/round3/round3_cost.png)

F1 medio (± desviación) frente al coste de inferencia, al tamaño y al tiempo de entrenamiento, en escala logarítmica. Un modelo compensa si está más arriba sin irse mucho a la derecha.

### Velocidad y memoria

![Ronda 3 · Velocidad y memoria](figures/round3/round3_speed.png)

Medidas en la RTX 4060 (la línea base, en CPU). Los tokens por segundo solo existen en el transformer y el híbrido.

### Curvas de aprendizaje

![Ronda 3 · Curvas de aprendizaje](figures/round3/round3_curves.png)

Media (y banda de ± 1 desviación) de los entrenamientos finales, época a época. Si la pérdida de entrenamiento baja mientras la de validación sube, el modelo está memorizando: es el sobreajuste.

### Brecha de generalización

![Ronda 3 · Brecha de generalización](figures/round3/round3_generalisation.png)

F1 del modelo final (media de sus semillas) en entrenamiento, validación y test. La distancia entre el círculo y el rombo es lo que el modelo memoriza y no generaliza.

### De dónde viene la variación

![Ronda 3 · De dónde viene la variación](figures/round3/round3_variance.png)

Desviación del F1 de test entre particiones (con la media de semillas) y entre semillas dentro de una misma partición. Es el análisis de Bouthillier et al. (2021): justifica usar muchas particiones antes que muchas semillas.

### Elección de configuración

![Ronda 3 · Elección de configuración](figures/round3/round3_selection.png)

F1 de validación (media ± desviación de las 25 particiones) de cada configuración; la rellena es la elegida y el rombo su F1 de test. Si las barras se solapan mucho, la elección es casi un empate.

### Matrices de confusión conjuntas

![Ronda 3 · Matrices de confusión conjuntas](figures/round3/round3_confusion.png)

Todas las predicciones de test de las 25 particiones juntas (cada segmento cuenta 5 veces, una por repetición). Filas: clase real; columnas: clase predicha.

### Acierto por buque

![Ronda 3 · Acierto por buque](figures/round3/round3_vessels.png)

Fracción de las veces que cada buque estuvo en test en que sus grabaciones se clasificaron bien. Muestra los buques que todos los modelos fallan.

### Acuerdo entre modelos

![Ronda 3 · Acuerdo entre modelos](figures/round3/round3_agreement.png)

Para cada pareja de modelos, en qué fracción de las grabaciones de test predicen lo mismo y en qué fracción fallan los dos. Si dos modelos fallan en sitios distintos, combinarlos podría ayudar.

### Calibración

![Ronda 3 · Calibración](figures/round3/round3_calibration.png)

Si el modelo dice "80 % seguro", ¿acierta el 80 % de las veces? La diagonal es la calibración perfecta; por debajo, el modelo está demasiado seguro. ECE y Brier: cuanto más bajos, mejor.

### Qué frecuencias usa cada modelo

![Ronda 3 · Qué frecuencias usa cada modelo](figures/round3/round3_interpretability.png)

Arriba, cuánto depende cada modelo de cada banda de 40 Hz (normalizado por fila), y en la última fila cuánto difiere esa banda entre las grabaciones de 2016 y 2017. Si un modelo usa las bandas que delatan la época, puede estar aprendiendo la época y no el buque. Abajo, a qué bandas atiende el token de clase del transformer y del híbrido.

### ROC de clasificación

![Ronda 3 · ROC de clasificación](figures/round3/round3_roc.png)

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

