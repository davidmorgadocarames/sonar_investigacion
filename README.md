# sonar_investigacion

Pipeline de procesado de señal sonar pasivo sobre el dataset DeepShip (4 clases: Cargo,
Passengership, Tanker, Tug):

```
WAV → control de calidad → preprocesado → LOFAR y DEMON → clasificador → evaluación → informe
```

**Resultado en una frase:** solo una línea base sencilla supera al azar con todo su intervalo
(F1 macro 0,374, IC 0,27-0,47, frente a 0,238), y dentro de cada época de grabación ningún modelo
lo supera, así que con estos datos no se puede afirmar que se reconozca la firma del buque.
Informe completo: [docs/report.md](docs/report.md).

## En gráficas

Las figuras principales, con lo necesario para leerlas. Las cifras completas están en el informe.

### 1. Datos

![Minutos por clase y partición](docs/figures/data_minutes.png)

**Las cuatro clases de buque:**

- **Cargo:** buque de carga (portacontenedores, graneleros). Grande, con motor diésel lento y
  hélice grande.
- **Passengership:** buque de pasaje; en este subconjunto son sobre todo transbordadores (ferris).
- **Tanker:** petrolero o buque cisterna, que transporta líquidos a granel.
- **Tug:** remolcador. Pequeño pero muy potente; empuja o arrastra a otros barcos.

**Entrenamiento, validación y test.** Con *entrenamiento* aprende el modelo; con *validación* se
eligen sus ajustes; *test* se reserva para medir el resultado final sin haberlo usado antes. La
partición se hace **por buque**: si el mismo barco estuviera en entrenamiento y en test, el
modelo podría reconocer *ese barco concreto* en vez de su tipo, y el resultado saldría inflado.

**Cómo se lee.** Cada barra son minutos de audio; encima, `b` es el número de buques y `g` el de
grabaciones. Destacan el desequilibrio (Cargo tiene mucho más audio que el resto) y que Tug solo
tiene 3 buques, uno en cada partición.

### 2. Señal: qué son LOFAR y DEMON

Un sonar pasivo no emite nada: solo escucha con un hidrófono el ruido que radia el barco. Ese
ruido lleva dos tipos de información, y cada técnica extrae uno:

- **LOFAR** (*Low Frequency Analysis and Recording*) muestra qué frecuencias suenan y cómo
  evolucionan en el tiempo. Sirve para ver las **líneas tonales**: sonidos de frecuencia fija que
  produce la maquinaria (motor, generadores, bombas).
- **DEMON** (*Detection of Envelope Modulation On Noise*): la hélice produce un ruido ancho por
  **cavitación** (burbujas que colapsan), y ese ruido "late" al ritmo de la hélice. DEMON mide a
  qué ritmo late, lo que revela la velocidad del eje y el número de palas.

![Validación LOFAR](docs/figures/signal/validation_lofar.png)
![Validación DEMON](docs/figures/signal/validation_demon.png)

**Qué significan 440 Hz y 12 Hz.** Son pruebas con señales sintéticas cuya respuesta se conoce de
antemano. Un tono de 440 Hz (la nota *la*) escondido en ruido debe aparecer como un pico justo
en 440 Hz en el LOFAR, lo que comprueba que el eje de frecuencias es correcto. Un ruido que
"late" 12 veces por segundo, como lo haría una hélice, debe dar un pico en 12 Hz en el DEMON. La
línea discontinua marca la frecuencia conocida: si el pico cae sobre ella, el procesado es
fiable.

### 3. Un buque real

![Ejemplo Passengership](docs/figures/signal/example_passengership.png)

Grabación de entrenamiento de un Passengership (el MALASPINA).

- **LOFARgram (arriba).** En horizontal, la frecuencia (0 a 2000 Hz); en vertical, el tiempo
  hacia abajo; el color indica cuánto sobresale cada frecuencia del ruido de fondo, en dB. Las
  **líneas verticales** son líneas tonales: frecuencias que se mantienen en el tiempo (aquí,
  hacia 80, 594 y 682 Hz) y que funcionan como una "huella" de la maquinaria del barco.
- **DEMON (abajo).** En horizontal, la frecuencia de modulación (0 a 100 Hz). El pico hacia
  2,8-3 Hz y su doble (5,7 Hz) son el **ritmo del eje** (unas 170-180 vueltas por minuto). La
  serie cada 11,3 Hz (11,3, 22,7, 34, 45,3...) es el **paso de pala** y sus **armónicos**: un
  armónico es un múltiplo entero de una frecuencia base. Como 11,3 / 2,8 ≈ 4, sugiere una hélice
  de **4 palas**.

Estos son los rasgos que un clasificador debería aprender para distinguir buques: qué líneas
tonales tiene cada barco y a qué ritmo late su hélice.

### 4. Resultados en test

![F1 con intervalos](docs/figures/eval2/results_f1.png)

**Los tres modelos usan exactamente los mismos datos** (los mismos segmentos de 3 s y la misma
partición); solo cambia el modelo y cómo se entrena:

- **Línea base:** una regresión logística, el modelo más simple. Ve un resumen de cada segmento
  (el LOFAR promediado en el tiempo más el DEMON: 1302 números) y aprende un peso por frecuencia.
  Sirve de referencia.
- **CNN sin aumento:** una red neuronal convolucional que ve el LOFARgram completo (11 instantes
  × 1001 frecuencias) y el DEMON, y puede aprender patrones más complejos.
- **CNN con aumento:** la misma red, pero durante el entrenamiento cada segmento se altera al azar
  (se tapan bandas de frecuencia o instantes y se desplaza un poco en el tiempo) para que no lo
  memorice. Validación y test nunca se alteran.

**Ronda 1 y ronda 2.** En la ronda 1 se entrenaron los modelos y se evaluó test por primera vez.
Después, una revisión del código encontró fallos de implementación y de medida (por ejemplo, dos
de las ocho salidas de la rama DEMON de la CNN valían siempre cero). Se arreglaron **sin cambiar
nada para mejorar la cifra**, se reentrenaron las dos CNN y se evaluó test una segunda y última
vez: es la **ronda 2**, la de esta gráfica. La línea base no cambió.

**Cómo se lee.** El *F1 macro* va de 0 a 1 (1 es acertar todo) y pondera igual las cuatro clases.
Cada punto lleva su **intervalo del 95 %** (la barra vertical) y la línea discontinua es el
**azar medido** (lo que sacaría un clasificador que responde al azar). El grupo de la izquierda
usa las 4 clases; los otros dos comparan clases **grabadas en la misma época** (Cargo frente a
Tug, de 2017; Passengership frente a Tanker, de 2016), que es lo que de verdad mide la firma del
buque, porque en esas parejas las condiciones de grabación son las mismas.

**Conclusión.** Solo la línea base tiene todo su intervalo por encima del azar con 4 clases, y
dentro de cada época ningún modelo lo supera.

### 5. Matrices de confusión

![Matrices de confusión](docs/figures/eval2/results_confusion.png)

**Cómo se leen.** Las **filas** son la clase real y las **columnas**, la clase que predijo el
modelo. La **diagonal** son los aciertos. Cada número es qué fracción de esa clase real acabó en
esa columna (entre paréntesis, el recuento). Arriba, **por segmento** (cada trozo de 3 s); abajo,
**por grabación** (la media de todos sus segmentos).

**Por qué la CNN sin aumento lo predice casi todo como Passengership y la CNN con aumento casi
todo como Cargo.** Es un síntoma de que **no generalizan**, no una preferencia real por esas
clases. Con solo 43 buques, la red **memoriza los buques de entrenamiento** (su mejor resultado en
validación llega hacia la época 3 y después solo memoriza); ante buques que nunca ha visto, sus
salidas **se colapsan hacia una sola clase**. Qué clase "gana" no tiene un significado físico:
depende del azar del entrenamiento (la semilla y el aumento), y por eso cambia de una CNN a otra.
Es una explicación razonada a partir de las curvas de entrenamiento, no algo demostrado con un
experimento específico.

La **línea base** acierta sobre todo Cargo y Tanker, y confunde Tug con Cargo, dos clases grabadas en la
misma época, lo que encaja con la limitación de la época.

### 6. Qué frecuencias usa cada modelo

![Interpretabilidad por bandas](docs/figures/eval2/interpretability_bands.png)

**Eje horizontal:** bandas de 40 Hz del LOFAR, de 0 a 2000 Hz. **Eje vertical, panel a panel:**

1. **Línea base:** cuánto **baja el F1** si se desordena esa banda entre segmentos (se rompe su
   relación con la clase). Cuanto más alta la barra, más depende el modelo de esa banda; los
   valores negativos o cercanos a cero indican que la banda no le sirve.
2. y 3. **Las dos CNN:** cuánto **baja la probabilidad de acertar la clase real** si se tapa esa
   banda (se rellena con el valor medio).
4. **Diferencia entre épocas:** cuánto difiere el **nivel medio de cada banda entre las
   grabaciones de 2016 y las de 2017**, en dB. Una barra alta indica una banda que delata la época
   de grabación.

**Qué se busca y qué se ve.** Si las bandas que usa un modelo coinciden con las que delatan la
época, el modelo podría estar aprendiendo la época y no el buque. La línea base usa sobre todo
40-80, 1240-1280 y 1440-1480 Hz, y **las dos últimas están entre las que más difieren entre
épocas**. Las CNN dependen sobre todo de **0-40 Hz**, frecuencias muy bajas donde dominan el ruido
del mar y de la plataforma de grabación más que la firma del buque. Todas las importancias son
pequeñas: ninguna banda decide por sí sola.

## Requisitos

- git
- Docker con Docker Compose v2 (en Windows y macOS, Docker Desktop)

No hace falta instalar Python ni ninguna librería: todo se ejecuta dentro del contenedor, con
Python 3.11.17 y versiones fijadas de cada librería. Lo que tengas instalado en tu ordenador no
se usa, así que no necesitas un entorno virtual ni preocuparte por las versiones (D-021).

## Puesta en marcha

1. Clona el repositorio (es privado; necesitas que David te dé acceso):

   ```bash
   git clone https://github.com/davidmorgadocarames/sonar_investigacion.git
   cd sonar_investigacion
   ```

2. Elige una opción **una sola vez**:

   - **Opción A · CPU (por defecto).** No hagas nada: funciona en cualquier ordenador. Es todo lo
     que necesitas para evaluar con los pesos entrenados.
   - **Opción B · GPU NVIDIA (solo para reentrenar más rápido).** Necesitas una GPU NVIDIA de la
     serie RTX 20 / GTX 16 o posterior, con el driver 580 o más nuevo, y Docker Desktop con WSL2
     en Windows (o el NVIDIA Container Toolkit en Linux). Crea en la carpeta
     `sonar_investigacion` un fichero llamado `.env` con estas dos líneas:

     ```
     COMPOSE_PATH_SEPARATOR=:
     COMPOSE_FILE=compose.yaml:compose.gpu.yaml
     ```

     A partir de ahí, todos los comandos de este README son los mismos y usan tu GPU. La imagen
     de la opción B ocupa unos 9,4 GB. Si tu driver es demasiado antiguo, el contenedor fallará al
     arrancar: actualiza el driver o borra `.env` para volver a la opción A.

3. **Solo en Linux:** para que los ficheros que escribe el contenedor sean tuyos, añade tu
   usuario al fichero `.env` (se crea si no existe). En Windows y macOS no hace falta.

   ```bash
   printf '\nHOST_UID=%s\nHOST_GID=%s\n' "$(id -u)" "$(id -g)" >> .env
   ```

   El salto de línea inicial evita pegarse a la última línea del `.env` de la opción B.

   **Con Docker sin root (*rootless*) o con Podman** el usuario `root` del contenedor ya
   corresponde a tu usuario del host: en ese caso pon en `.env` `HOST_UID=0` y `HOST_GID=0`
   (comprobado que la imagen funciona con el UID 0; sin probar en un Docker *rootless* real).

4. Construye la imagen. La primera vez tarda unos minutos; la imagen de la opción A ocupa unos
   1,9 GB:

   ```bash
   docker compose build
   ```

5. Comprueba que funciona:

   ```bash
   docker compose run --rm pipeline --help
   ```

   Debe mostrar los comandos `qc`, `features`, `train`, `evaluate` y `report`.

## Comprobar la instalación

```bash
docker compose run --rm --entrypoint python pipeline -m pytest -rs
```

Tarda alrededor de un minuto. Debe terminar con todos los tests en verde (`passed`). La opción `-rs`
lista los tests saltados y por qué: sin los datos descargados se saltan los que los necesitan
(dicen `not found` o `without WAVs`, y los que necesitan los rasgos dicen `has no features: run
features first`). En la opción A se salta el de determinismo en CUDA (dice `no CUDA here`) y
en la opción B el que comprueba el error sin GPU (dice `this machine has CUDA`). Saltarse no es
un error.

## Datos

Los audios **no están en este repositorio**. Se descargan del repositorio público de los
autores de DeepShip, fijado en una versión concreta para que todos usemos exactamente los mismos
ficheros. Licencia, contenido y detalles: [datasets/README.md](datasets/README.md).

Necesitas unos **1,2 GB** libres: unos 700 MB de audio y unos 500 MB del historial de git de
DeepShip (la carpeta `.git`), que es lo que permite comprobar que los ficheros están íntegros.
Desde la carpeta `sonar_investigacion`, ejecuta estos dos
comandos (el primero tarda alrededor de un minuto, según tu conexión):

```bash
git clone -c core.autocrlf=false https://github.com/irfankamboh/DeepShip.git datasets/raw/DeepShip
git -C datasets/raw/DeepShip checkout 77725726a33d6aef4ce1d0fe70f67c0bbee4b5b6
```

El segundo comando muestra un aviso de *detached HEAD*. Es normal: significa que estás en esa
versión fija.

Para comprobar que la descarga es correcta:

```bash
git -C datasets/raw/DeepShip status
```

Debe decir `HEAD detached at 7772572` y `nothing to commit, working tree clean`. Eso garantiza
que los ficheros son idénticos a los de la versión fijada. Deben quedar 63 ficheros `.wav`
repartidos en `Cargo/` (12), `Passengership/` (20), `Tanker/` (28) y `Tug/` (3).

Si ya lo habías descargado, no hace falta repetirlo.

## Control de calidad y rasgos

Con los datos descargados, desde la carpeta `sonar_investigacion`:

```bash
docker compose run --rm pipeline qc
docker compose run --rm pipeline features
```

- `qc` revisa los 63 WAV (formato, duración, offset DC, saturación...) en unos segundos y escribe
  `outputs/qc/qc.csv`. No modifica ni excluye ningún fichero. Es normal que marque `flag_dc` en
  casi todas las grabaciones: todas tienen un offset DC, y `features` lo corrige (D-009).
- `features` tarda menos de un minuto. Comprueba que cada WAV es idéntico al de la versión
  fijada (SHA-256 de `datasets/manifests/vessels.csv`), quita el offset DC de cada grabación, la
  corta en segmentos de 3 s y calcula el LOFAR (líneas tonales, 0 a 2 kHz) y el DEMON (modulación
  de la cavitación, 0 a 100 Hz) de cada segmento. Debe decir `63 recordings, 1889 segments` y
  escribir en `datasets/processed/` unos 82 MB: un `features_<partición>.npz` por partición
  (train, val y test), `index.csv` (una fila por segmento) y `features_meta.json` (parámetros
  usados). Esa carpeta no se sube a git.

Figuras de ejemplo (en `outputs/figures/`, que no se sube a git):

```bash
docker compose run --rm pipeline features --figures
docker compose run --rm --entrypoint python pipeline -m pytest tests/test_figures.py --figures-dir outputs/figures
```

- El primero calcula los rasgos y dibuja, para la grabación más larga de cada clase dentro de
  la partición de entrenamiento (para no mirar val ni test), su LOFARgram y su DEMON medio:
  `example_<clase>.png`.
- El segundo dibuja la validación con señales sintéticas: un tono de 440 Hz debe salir en
  440 Hz en el LOFAR (`validation_lofar.png`), y un ruido modulado a 12 Hz debe dar un pico en
  12 Hz en el DEMON (`validation_demon.png`).

## Evaluar con los pesos entrenados

Los pesos de los tres modelos están en `models/` (en total menos de 1 MB). Con los rasgos ya
calculados (`features`, sección anterior):

```bash
docker compose run --rm pipeline evaluate
```

Tarda unos 30 segundos en CPU. Puntúa la partición de test con los tres modelos y escribe
`outputs/results/metrics.json` y las figuras `outputs/figures/results_*.png`. Debe mostrar estas
cifras de F1 macro por segmento, idénticas a las del informe (`docs/results/eval2.json`). La
reproducción exacta está comprobada en x86-64 (ver "Equipo de referencia"); en otros
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
código cambió después de la evaluación 2 sin afectar a las cifras (detalle en
[docs/results/README.md](docs/results/README.md)). Las métricas y los `weights_sha256` deben
coincidir.

Otras opciones de `evaluate`:

- `--plot-only`: redibuja las figuras desde `metrics.json` sin volver a puntuar.
- `--interpretability`: qué bandas de frecuencia usa cada modelo (sobre val, unos 30 s).

Para regenerar el informe desde los resultados versionados (no hace falta haber descargado los
datos):

```bash
docker compose run --rm pipeline report
```

## Reentrenar (opcional)

Reentrenar **sobrescribe los pesos de `models/`**. Para volver a los versionados:
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

## Equipo de referencia

Todos los tiempos y resultados de este README se midieron en:

| Componente | Detalle |
|---|---|
| CPU | Intel Core i7-14700KF (28 hilos), x86-64 |
| GPU | NVIDIA GeForce RTX 4060, 8 GB (driver 610.62; PyTorch 2.14.1 con CUDA 13.0) |
| Sistema | Windows 10 Pro con Docker Desktop (WSL2) |
| Imágenes | `sonar-pipeline:latest` (CPU, 1,9 GB) y `sonar-pipeline:gpu` (9,4 GB) |

## Limitaciones

Están explicadas en el informe ([docs/report.md](docs/report.md), sección 5). En resumen:

- En este subconjunto de DeepShip la clase coincide con la época de grabación (Cargo y Tug en
  2017, Passengership y Tanker en 2016): el resultado de 4 clases no distingue el tipo de buque
  de las condiciones de grabación.
- Hay pocos buques (43); Tug tiene 3, con una sola grabación en test, y su resultado no es fiable.
- La ROC es de clasificación, no de detección: DeepShip no tiene grabaciones sin buque.
- Test se evaluó dos veces (la segunda tras arreglar fallos de la primera revisión); el detalle y
  qué código produjo cada resultado están en [docs/results/README.md](docs/results/README.md).

## Carpetas

| Carpeta | Qué contiene |
|---|---|
| `datasets/` | Manifiestos versionados (`manifests/`); audios y rasgos, no (`raw/`, `processed/`). |
| `models/` | Pesos entrenados y su configuración, versionados. |
| `docs/` | Informe, resúmenes de resultados y figuras, versionados (montada en el contenedor para `report`). |
| `outputs/` | Resultados y figuras de cada ejecución. No se suben a git. |
| `src/sonar_pipeline/` | Código de la pipeline. |
| `tests/`, `tools/` | Tests y scripts de apoyo. |

## Licencia

El código y la documentación están bajo la licencia MIT ([LICENSE](LICENSE)), © 2026 David
Morgado Carames. No cubre los datos de DeepShip (no se distribuyen aquí) ni los pesos de
`models/`, que derivan de DeepShip y siguen su mismo supuesto de uso académico (D-022).

## Cita

Datos: Irfan, M. et al. (2021). *DeepShip: An underwater acoustic benchmark dataset and a
separability based deep learning framework for ship classification.* Expert Systems with
Applications, 183, 115270.

DeepShip no tiene licencia explícita: se asume que puede usarse con fines académicos citando a
los autores (supuesto no confirmado por ellos), y ningún audio se redistribuye. Detalles en
[datasets/README.md](datasets/README.md).
