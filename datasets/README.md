# Datos

## DeepShip

- **Contenido:** grabaciones de ruido radiado por buques, en 4 clases: Cargo, Passengership,
  Tanker y Tug.
- **Fuente:** https://github.com/irfankamboh/DeepShip, último commit
  `77725726a33d6aef4ce1d0fe70f67c0bbee4b5b6` (2023-03-23), revisado el 2026-10-10.
  - Es un **subconjunto** del dataset: 63 WAV, unos 733 MB y 1,59 h en total (Cargo 12,
    Passengership 20, Tanker 28, Tug 3), de 44 buques distintos según
    los metafiles (43 grupos de partición tras D-011).
  - Formato de todos los WAV: 32 kHz, mono, float32. El módulo `wave` de Python no puede
    leerlos ([D-008](../DECISIONS.md)).
  - Cada carpeta de clase tiene además un `*-metafile`: un CSV sin cabecera con una fila por
    grabación del dataset completo (`id, código AIS, buque, fecha, hora, duración_s, ?`). El
    significado del último campo no está documentado. La relación entre WAV y fila no es
    siempre por id ([D-007](../DECISIONS.md), [DT-002](../DECISIONS.md)).
  - El dataset completo (47 h y 265 buques, según el README de la fuente) se pide por correo a
    los autores. Este proyecto no lo usa.
- **Licencia:** **sin licencia explícita.** El repositorio no tiene fichero LICENSE y su README
  no dice nada sobre el uso. Se asume que puede usarse con fines académicos citando a los
  autores; es un supuesto de David que los autores no han confirmado. **No se redistribuye**:
  ningún audio de DeepShip se sube a este repositorio ni a ningún otro sitio.
- **Cita:** Irfan, M. et al. (2021). *DeepShip: An underwater acoustic benchmark dataset and a
  separability based deep learning framework for ship classification.* Expert Systems with
  Applications, 183, 115270.
- **Alcance:** solo se usa este subconjunto, para entrenar y para evaluar
  ([D-005](../DECISIONS.md)).
- **Cómo obtenerlo:** con los comandos git del [README](../README.md), que clonan la fuente en
  `raw/DeepShip` y la fijan en el commit indicado arriba ([D-006](../DECISIONS.md)).
- **Riesgo conocido:** si los autores borran el repositorio o reescriben su historial, la
  descarga deja de funcionar.

## Estructura

| Carpeta | En git | Contenido |
|---|---|---|
| `raw/` | No | Audios originales, tal como se descargan. |
| `processed/` | No | Datos derivados (segmentos, rasgos). Se regeneran con la pipeline. |
| `manifests/` | **Sí** | Inventario, checksums y partición train/val/test. |

Ningún audio entra en git sin aprobación expresa: lo impiden `.gitignore` y el hook
`forbid-audio` de pre-commit.

## Tabla WAV → buque (`manifests/vessels.csv`)

Una fila por WAV (63), ordenadas por clase y número de WAV; UTF-8 con finales de línea LF. La
genera `tools/build_vessel_table.py` a partir de los WAV y los metafiles
([D-007](../DECISIONS.md), [DT-002](../DECISIONS.md)):

```bash
docker compose run --rm --entrypoint python pipeline tools/build_vessel_table.py
```

Regenerarla con los mismos datos da un fichero idéntico byte a byte.

- **Columnas:** `wav_path` (relativa a `raw/DeepShip/`), `vessel_class`, `wav_number`, `sha256`,
  `sample_rate`, `n_samples`, `wav_duration_s`; la fila del metafile usada (`metafile`,
  `metafile_line` (línea física, contando desde 1), `metafile_block`, `metafile_id`, `ais_code`,
  `vessel_name_raw` (tal cual, con sus espacios), `date`, `time`, `metafile_duration_s`, `field7`
  (sin interpretar)); `vessel_group`; y `match_source`, `duration_match`, `duration_diff_s`.
- **Cómo se asigna cada WAV a una fila.** La numeración del metafile vuelve a empezar en cada
  *bloque* (Tanker tiene 3). En cada clase, los WAV (por número) se alinean con las filas de un
  bloque conservando el orden de ambos, que es cronológico. Solo se emparejan si la duración
  coincide con ±1 s, y se maximiza, por este orden: el número de parejas, cuántas tienen
  id = número de WAV y la cercanía entre id y número. La clase usa el bloque que mejor se alinea.
  El script falla si el óptimo no es único o si una fila se usa dos veces.
- **`match_source`:** `id` (id = número y la duración coincide: 35), `duration` (la duración
  coincide con otra fila: 26) o `id_unverified` (ninguna duración coincide y se toma la fila por
  id: 2, Passengership 30 y 37, ambas COLUMBIA).
- **`vessel_group`:** clase + nombre normalizado (sin espacios a los lados, en mayúsculas y con
  los espacios internos colapsados). Todos los nombres que empiezan por "QUEEN OF" forman un solo
  grupo, al que también se une BOWEN QUEEN por precaución ([D-011](../DECISIONS.md)). Ningún
  grupo abarca dos clases. Resultado: 43 grupos (Cargo 11, Passengership 10, Tanker 19, Tug 3).

## Partición

La partición train/val/test se hace **por buque**, que los metafiles permiten
([D-007](../DECISIONS.md)). Nunca por segmento ni por grabación, para evitar fugas de datos. El
manifiesto con la partición se versiona en `manifests/`.

### Formato de `manifests/split.csv`

Lo genera F1-T06c. Una fila por WAV (los 63 de `vessels.csv`), ordenadas como `vessels.csv`
(clase y número de WAV); UTF-8 con finales de línea LF. Columnas obligatorias (puede haber más):

| Columna | Contenido |
|---|---|
| `wav_path` | Ruta relativa a `raw/DeepShip/`, igual que en `vessels.csv` y `qc.csv`. |
| `vessel_class` | Cargo, Passengership, Tanker o Tug. |
| `vessel_group` | Grupo de partición (buque), copiado de `vessels.csv`. |
| `split` | `train`, `val` o `test`. |

`vessels.csv` es la referencia: el SHA-256, la duración, la clase y el grupo se toman de ahí.

### Comprobación de fugas (`tools/check_leaks.py`)

```bash
docker compose run --rm --entrypoint python pipeline tools/check_leaks.py [manifests/split.csv]
```

Sale con código 1 y la lista de problemas si un grupo (`vessel_group`, el declarado o el de
`vessels.csv`) está en dos particiones; si un WAV aparece más de una vez o dos rutas tienen el
mismo SHA-256 (la misma grabación); si la clase o el grupo no coinciden con `vessels.csv`; si
falta o sobra algún WAV respecto a `vessels.csv`; o si un valor de `split` no es válido. Sale
con 2 si falta el fichero o una columna. Si todo está bien, sale con 0 y resume los WAV, grupos
y horas por partición y clase.

`--similar-names` lista los pares de grupos con WAV cuyos nombres se parecen, dentro de una
clase y entre clases ([DT-003](../DECISIONS.md)): uno es prefijo del otro, comparten alguna
palabra o están a distancia de edición ≤ 3. Es un listado para revisar a mano; no fusiona nada.

Los tests de `tests/test_vessel_table_data.py` comprueban [DT-002](../DECISIONS.md): en las 61
filas verificadas (`match_source` `id` o `duration`) la duración del WAV coincide con la de su
fila del metafile (±1 s), y solo Passengership 30 y 37 quedan sin verificar, ambas en COLUMBIA.
Si existe `raw/DeepShip`, además leen cada WAV y su línea del metafile y comprueban que la
duración, el número de muestras y el SHA-256 coinciden con la tabla; si no existe, esos tests
se saltan (`skip`).
