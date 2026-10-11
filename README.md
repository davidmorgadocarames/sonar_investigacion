# sonar_investigacion

Pipeline de procesado de señal sonar pasivo sobre el dataset DeepShip (4 clases: Cargo,
Passengership, Tanker, Tug):

```
WAV → control de calidad → preprocesado → LOFAR y DEMON → clasificador → evaluación → informe
```

**Resultado en una frase:** solo una línea base sencilla supera al azar (F1 macro 0,374 frente a
0,238), y dentro de cada época de grabación ningún modelo se distingue del azar, así que con estos
datos no se puede afirmar que se reconozca la firma del buque. Informe completo:
[docs/report.md](docs/report.md).

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
   echo "HOST_UID=$(id -u)" >> .env
   echo "HOST_GID=$(id -g)" >> .env
   ```

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
features first`), y en la opción A se salta el de determinismo en CUDA (dice `no CUDA here`).
Saltarse no es un error.

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
  escribir en `datasets/processed/` unos 85 MB: un `features_<partición>.npz` por partición
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
cifras de F1 macro por segmento, idénticas a las del informe (`docs/results/eval2.json`):

| Modelo | Debe salir |
|---|---|
| `baseline` | `segment F1 0.374 (0.27-0.47)` |
| `cnn` | `segment F1 0.115 (0.07-0.16)` |
| `cnn_aug` | `segment F1 0.252 (0.17-0.34)` |

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
| `docker compose run --rm pipeline train --model cnn` | CNN: 4 configuraciones y 3 semillas | ~2-3 min (estimado) | ~40 s |
| `docker compose run --rm pipeline train --model cnn-aug` | CNN con aumento de datos (necesita `cnn` antes) | ~1-2 min (estimado) | ~30 s |
| `docker compose run --rm pipeline train --model baseline --permutation-test 200` | Test de permutación de la línea base | ~18 min | (usa CPU) |

- La línea base y la comprobación de duración dan exactamente los mismos resultados en cualquier
  máquina.
- Las CNN son deterministas en el mismo tipo de dispositivo, pero los pesos versionados se
  entrenaron en GPU (una RTX 4060): reentrenar en CPU, o en otra GPU, da pesos y cifras algo
  distintos.

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
| `docs/` | Informe, resúmenes de resultados y figuras, versionados. |
| `outputs/` | Resultados y figuras de cada ejecución. No se suben a git. |
| `src/sonar_pipeline/` | Código de la pipeline. |
| `tests/`, `tools/` | Tests y scripts de apoyo. |

## Cita

Datos: Irfan, M. et al. (2021). *DeepShip: An underwater acoustic benchmark dataset and a
separability based deep learning framework for ship classification.* Expert Systems with
Applications, 183, 115270.

DeepShip no tiene licencia explícita: se asume que puede usarse con fines académicos citando a
los autores (supuesto no confirmado por ellos), y ningún audio se redistribuye. Detalles en
[datasets/README.md](datasets/README.md).
