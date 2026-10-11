# sonar_investigacion

Pipeline de procesado de señal sonar pasivo sobre el dataset DeepShip (4 clases: Cargo,
Passengership, Tanker, Tug):

```
WAV → control de calidad → preprocesado → LOFAR y DEMON → clasificador → evaluación → informe
```

> **Estado:** en construcción. Este README se completa en la fase F4. Por ahora se puede
> construir la imagen, descargar los datos, ejecutar el control de calidad (`qc`) y calcular
> los rasgos LOFAR y DEMON (`features`).

## Requisitos

- git
- Docker con Docker Compose v2 (en Windows y macOS, Docker Desktop)

No hace falta instalar Python ni nada más: todo se ejecuta dentro del contenedor.

## Puesta en marcha

1. Clona el repositorio (es privado; necesitas que David te dé acceso):

   ```bash
   git clone https://github.com/davidmorgadocarames/sonar_investigacion.git
   cd sonar_investigacion
   ```

2. Construye la imagen. La primera vez tarda unos minutos; la imagen ocupa unos 500 MB:

   ```bash
   docker compose build
   ```

3. Comprueba que funciona:

   ```bash
   docker compose run --rm pipeline --help
   ```

   Debe mostrar los comandos `qc`, `features`, `train`, `evaluate` y `report`. Por ahora
   están implementados `qc` y `features`; los demás llegan en fases posteriores.

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
  `outputs/qc/qc.csv`. No modifica ni excluye ningún fichero.
- `features` tarda menos de un minuto. Quita el offset DC de cada grabación, la corta en
  segmentos de 3 s y calcula el LOFAR (líneas tonales, 0 a 2 kHz) y el DEMON (modulación de la
  cavitación, 0 a 100 Hz) de cada segmento. Debe decir `63 recordings, 1889 segments` y escribir
  en `datasets/processed/` unos 85 MB: un `features_<partición>.npz` por partición (train, val y
  test), `index.csv` (una fila por segmento) y `features_meta.json` (parámetros usados). Esa
  carpeta no se sube a git.

## Evaluación con los pesos entrenados

*Pendiente (fase F4).*

## Reentrenar (opcional)

*Pendiente (fase F4).*

## Carpetas

| Carpeta | Qué contiene |
|---|---|
| `datasets/` | Datos. Los audios no se suben a git. |
| `models/` | Pesos entrenados. |
| `outputs/` | Resultados y figuras. No se suben a git. |
