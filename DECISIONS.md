# DECISIONS.md

Registro de decisiones y deuda técnica. **Solo se apunta lo que David ha aprobado en el chat**
(regla 2 de [CLAUDE.md](CLAUDE.md)). Los IDs no se reutilizan. Una decisión que cambia otra
anterior se apunta como entrada nueva que la sustituye, y la antigua no se borra.

## Decisiones

### D-001 · Docker con CPU por defecto
- **Fecha:** 2026-10-10
- **Contexto:** un tercero (el amigo de David) debe poder probar el proyecto en su casa con solo
  git y Docker, sin instalar Python ni drivers. El entrenamiento completo se hace en el servidor
  de David, que tiene GPU.
- **Alternativas:** (a) imagen con CUDA por defecto; (b) instalación local con Python y venv;
  (c) imagen CPU por defecto con un override opcional para GPU.
- **Decisión:** (c). `compose.yaml` usa CPU; `compose.gpu.yaml` es un override opcional para
  NVIDIA.
- **Motivo:** la evaluación con pesos ya entrenados no necesita GPU. Una imagen CPU funciona en
  cualquier máquina con Docker y ocupa mucho menos que una con CUDA.

### D-002 · Dependencias añadidas por fase
- **Fecha:** 2026-10-10
- **Contexto:** la regla 3 pide infraestructura mínima y la regla 6, versiones fijadas. En F0 no
  hay código de la pipeline.
- **Alternativas:** (a) fijar ya todo el stack previsto (numpy, scipy, soundfile, torch...);
  (b) añadir cada dependencia en la tarea que la necesita.
- **Decisión:** (b). En F0, `requirements.txt` solo contiene `pytest`.
- **Motivo:** la imagen es pequeña y rápida de construir mientras no hace falta más, y cada
  dependencia queda ligada a la tarea que la justifica.

### D-003 · Carpetas montadas versionadas con `.gitkeep`
- **Fecha:** 2026-10-10
- **Contexto:** `compose.yaml` monta `datasets/`, `models/` y `outputs/`. Si una carpeta montada
  no existe en el host, en Linux Docker la crea como root, y el usuario no root del contenedor
  no puede escribir en ella.
- **Alternativas:** (a) pedir en el README que se creen a mano; (b) versionarlas vacías con un
  `.gitkeep` y que `.gitignore` ignore todo lo demás que contengan.
- **Decisión:** (b), en `datasets/raw/`, `datasets/processed/`, `datasets/manifests/`,
  `models/`, `outputs/` y `tools/`.
- **Motivo:** las carpetas existen nada más clonar, sin pasos manuales, y su contenido sigue
  fuera de git.

### D-004 · Tests dentro de Docker
- **Fecha:** 2026-10-10
- **Contexto:** el amigo solo tiene git y Docker, y el entorno de referencia es la imagen.
- **Alternativas:** (a) tests con un venv local; (b) tests dentro del contenedor.
- **Decisión:** (b): `docker compose run --rm --entrypoint python pipeline -m pytest`. La imagen
  incluye `tests/` y `pytest`.
- **Motivo:** se prueba exactamente el entorno con el que se ejecuta la pipeline, sin depender
  del Python del host.

### D-005 · Alcance de los datos: solo el subconjunto público de DeepShip
- **Fecha:** 2026-10-11
- **Contexto:** David no tiene el dataset completo de DeepShip, que se pide por correo a los
  autores. El repositorio público tiene 67 grabaciones: Cargo 13, Passengership 21, Tanker 29 y
  Tug 4 (ver [datasets/README.md](datasets/README.md)).
- **Alternativas:** (a) pedir el dataset completo y esperar; (b) trabajar solo con el
  subconjunto público.
- **Decisión:** (b). El proyecto usa solo el subconjunto de
  https://github.com/irfankamboh/DeepShip, fijado en el commit
  `77725726a33d6aef4ce1d0fe70f67c0bbee4b5b6`, para entrenar y para evaluar.
- **Motivo:** está disponible ya y es suficiente para un ejercicio de principio a fin.
- **Consecuencias:** con 4 grabaciones de Tug, las métricas de esa clase serán muy inestables.
  La partición (F1-T06) garantizará al menos una grabación de cada clase en cada partición, y
  F3 dará intervalos por bootstrap. Si más adelante se consigue el dataset completo, se apuntará
  como una decisión nueva.
- **Corrección (2026-10-11):** el recuento original contaba el `*-metafile` de cada carpeta
  como si fuera una grabación. Son **63 WAV**: Cargo 12, Passengership 20, Tanker 28 y Tug 3
  (inventario F1-T03). La decisión no cambia.

### D-006 · Los datos le llegan al amigo con comandos git en el README
- **Fecha:** 2026-10-11
- **Contexto:** DeepShip no tiene licencia explícita y no se redistribuye (regla 4). El amigo
  solo tiene git y Docker.
- **Alternativas:** (a) subir los audios a este repositorio; (b) descarga manual; (c) script en
  `tools/` con verificación SHA-256; (d) comandos git en el README que clonan la fuente fijada
  en un commit.
- **Decisión:** (d). El README incluye los comandos para clonar `irfankamboh/DeepShip` en
  `datasets/raw/DeepShip` y fijarlo en el commit `77725726a33d6aef4ce1d0fe70f67c0bbee4b5b6`.
- **Motivo:** solo hace falta git, no se redistribuye nada y el checkout de un commit concreto
  garantiza que cada fichero es idéntico byte a byte, sin mantener checksums propios.
- **Riesgo aceptado:** si la fuente desaparece o se reescribe su historial, la descarga deja de
  funcionar y habrá que buscar otra fuente.
- **Ajuste (2026-10-11):** el `git clone` lleva `-c core.autocrlf=false`. Sin esa opción, en
  Windows git cambia los finales de línea del `README.txt` de la fuente (a los WAV no les
  afecta), y la promesa de ficheros idénticos byte a byte no se cumpliría.

### D-007 · Partición por buque
- **Fecha:** 2026-10-11
- **Contexto:** el inventario (F1-T03) muestra que los números de fichero identifican
  grabaciones, no buques. El nombre del buque está en el `*-metafile` de cada clase. Hay 44
  buques; algunos tienen varias grabaciones (KIRKEHOLMEN tiene 7 de las 28 de Tanker).
- **Alternativas:** (a) partir por grabación; (b) partir por buque.
- **Decisión:** (b). El grupo de partición es el nombre del buque dentro de su clase. La tabla
  que asigna cada WAV a su buque se versiona en `datasets/manifests/`, indicando si la fila se
  unió por id o se dedujo por duración. Los casos dudosos se juntan en un solo grupo:
  Passengership 30 y 37 van con COLUMBIA, y todos los "QUEEN OF" (nombre cortado) forman un
  único grupo.
- **Motivo:** partir por grabación pondría el mismo buque en train y en test (regla 5). Juntar
  grupos dudosos nunca provoca fugas; solo resta flexibilidad al repartir.
- **Consecuencia:** Tug tiene 3 buques, así que cada partición tendrá exactamente una grabación
  de Tug.

### D-008 · numpy y scipy para leer los WAV
- **Fecha:** 2026-10-11
- **Contexto:** todos los WAV son float32 y el módulo `wave` de la biblioteca estándar no los
  lee. El control de calidad (F1-T05) necesita leer las muestras.
- **Alternativas:** (a) `numpy` + `scipy.io.wavfile`; (b) `soundfile` (libsndfile); (c) `numpy`
  + un lector RIFF propio.
- **Decisión:** (a), con versiones fijadas en `requirements.txt`. Entran en F1-T05.
- **Motivo:** numpy y scipy harán falta igualmente en F2 para LOFAR y DEMON, así que no se añade
  ninguna dependencia que no se fuera a usar.

### D-009 · No se excluye ninguna grabación por el QC
- **Fecha:** 2026-10-11
- **Contexto:** el QC (F1-T05) marca un offset DC positivo casi constante (0,027 a 0,032) en los
  63 WAV, en la mayoría mayor que la señal. 16 grabaciones duran menos de 25 s (la más corta, 6 s:
  Passengership 30). No hay recortes, NaN ni silencio. La duración va ligada a la clase: todos
  los Cargo y Tug duran más de 120 s, y casi todos los Passengership y Tanker son cortos.
- **Alternativas:** (a) excluir los ficheros marcados (no quedaría ninguno); (b) excluir los que
  duren menos de un umbral (vaciaría Passengership y Tanker); (c) conservar los 63 y tratar los
  problemas en el preprocesado.
- **Decisión:** (c). El offset DC se resta por grabación en el preprocesado de F2; `flag_dc` se
  mantiene en el QC como información. No se excluye ninguna grabación corta: solo se exige que
  cada una tenga al menos un segmento completo. Si en F2 el segmento elegido dura más de 6 s, se
  revisa esta decisión para Passengership 30.
- **Motivo:** ningún fichero es defectuoso; el offset se corrige restando la media, y con tan
  pocos datos excluir recortaría las clases más pequeñas.

### D-010 · Asignación de cada WAV a su fila del metafile
- **Fecha:** 2026-10-11
- **Contexto:** la numeración de los metafiles tiene saltos, repeticiones y bloques que vuelven a
  empezar (Tanker tiene 3). Probar primero el id y después la duración no reproduce el inventario
  de F1-T03: en Tanker 33 y 35 cruza las filas y da 36/25/2 asignaciones en vez de 35/26/2 (el
  buque, KIRKEHOLMEN, es el mismo; cambian la fecha, la hora y la línea).
- **Alternativas:** (a) probar primero el id y después la duración; (b) reglas por rangos
  escritas a mano; (c) alineamiento que conserva el orden.
- **Decisión:** (c), implementada en `tools/build_vessel_table.py`. En cada clase, los WAV (por
  número) se alinean con las filas de un bloque del metafile conservando el orden de ambos, que
  es cronológico. Solo se emparejan si la duración coincide con ±1 s, y se maximiza, por este
  orden: el número de parejas, cuántas tienen id = número de WAV y la cercanía entre id y número.
  Un WAV sin pareja toma la fila de su id y se marca `id_unverified`. El script falla si el
  óptimo no es único, si una fila se usa dos veces o si una línea está mal formada. Las rutas de
  `vessels.csv` son relativas a `raw/DeepShip/`, igual que en `qc.csv`, para cruzar las dos
  tablas directamente.
- **Motivo:** se basa en una propiedad comprobable de los datos (el orden cronológico), no tiene
  excepciones a mano y, ante la duda, falla en vez de adivinar.

### D-011 · BOWEN QUEEN se une al grupo "QUEEN OF"
- **Fecha:** 2026-10-11
- **Contexto:** el listado de nombres parecidos de DT-003 (`tools/check_leaks.py
  --similar-names`) dio 11 pares; en casi todos la palabra común es la naviera o la serie de la
  flota. El único dudoso es BOWEN QUEEN frente a "QUEEN OF", un nombre cortado que puede ser
  cualquier "Queen of …" de BC Ferries.
- **Alternativas:** (a) mantenerlos separados; (b) juntarlos por precaución.
- **Decisión:** (b). `build_vessel_table.py` une BOWEN QUEEN al grupo `Passengership/QUEEN OF`
  (lista `MERGED_NAMES`). Quedan 43 grupos (Passengership pasa de 11 a 10). David revisó los demás
  pares y no son el mismo buque.
- **Motivo:** juntar grupos nunca provoca fugas; solo resta flexibilidad al repartir.

### D-012 · Formato del manifiesto de partición `split.csv`
- **Fecha:** 2026-10-11
- **Contexto:** `tools/check_leaks.py` (F1-T06b) comprueba la partición que generará F1-T06c.
- **Alternativas:** (a) columnas mínimas, tomando SHA-256 y duración de `vessels.csv`; (b) copiar
  también SHA-256 y duración en el split.
- **Decisión:** (a). `datasets/manifests/split.csv` con `wav_path` (relativa a `raw/DeepShip/`),
  `vessel_class`, `vessel_group` y `split` (`train`, `val` o `test`); una fila por WAV, en el orden
  de `vessels.csv`, UTF-8 y LF. La herramienta comprueba que clase y grupo coinciden con
  `vessels.csv`.
- **Motivo:** cada dato vive en un solo sitio, y el cruce con `vessels.csv` detecta cualquier
  desajuste.

### D-013 · Proporciones de la partición: 60/20/20 en minutos por clase
- **Fecha:** 2026-10-11
- **Contexto:** 43 grupos de buque y 95,4 min de audio (Cargo 11 grupos/38,7 min, Passengership
  10/19,1, Tanker 19/27,7, Tug 3/9,9). El grupo más grande pesa el 27 % de Passengership y el
  34 % de Tug.
- **Alternativas:** (a) 60/20/20; (b) 70/15/15 (val y test con 1 o 2 buques por clase); (c) sin
  val, con validación cruzada (se valorará en F3 dentro de train + val, sin tocar test).
- **Decisión:** (a), medido en **minutos de audio dentro de cada clase**, asignando buques
  enteros. Cada clase tiene al menos un buque en cada partición (Tug queda 1/1/1). Un algoritmo
  determinista con semilla fijada elige el reparto más cercano al objetivo, y
  `tools/check_leaks.py` lo valida.
- **Motivo:** los segmentos del modelo salen de los minutos, no de los ficheros, cuyas
  duraciones van de 6 s a 204 s. Con tan pocos buques el reparto será aproximado; la tabla real
  de minutos por partición y clase se revisa con David antes del commit.
- **Ajuste (2026-10-11):** `tools/make_split.py` hace una búsqueda exacta (ramificación y poda)
  sin azar, así que la semilla no interviene: el mismo `vessels.csv` da siempre el mismo
  `split.csv`. Los empates (varios repartos con el mismo coste, como en Tanker) se resuelven en
  un orden fijo: grupos por duración descendente y luego por nombre, probando train < val < test.
  David aprobó la partición resultante (SHA-256 `51564cf1…b5a7`).

### D-014 · `vessels.csv` versiona un extracto de los metafiles de DeepShip
- **Fecha:** 2026-10-11
- **Contexto:** la revisión de F1 (F1-T08) señaló que `datasets/manifests/vessels.csv` copia parte
  de los metafiles de DeepShip (nombre del buque, código AIS, fecha, hora, duración y el 7.º
  campo) para 63 grabaciones, y que la licencia de DeepShip no está confirmada (solo hay un
  supuesto de uso académico).
- **Alternativas:** (a) conservar todas las columnas; (b) quitar las que no hacen falta para
  partir (código AIS, fecha, hora y 7.º campo).
- **Decisión:** (a). `vessels.csv` conserva sus columnas y datos, bajo el mismo supuesto de
  licencia que D-006: uso académico con cita, en un repositorio privado. No se redistribuye ningún
  audio. El criterio de salida de F1 pasa a ser "licencia revisada y supuesto aceptado por David".
- **Motivo:** son datos factuales y pocos, y son necesarios para trazar cada asignación WAV →
  buque (D-010).

### D-015 · Parámetros de preprocesado, LOFAR, DEMON y formato de los rasgos
- **Fecha:** 2026-10-11
- **Contexto:** F2 implementa el preprocesado y los rasgos LOFAR y DEMON (F2-T04 a F2-T07). Los
  WAV son de 32 kHz y duran de 6 s a 204 s (95,4 min en total). Segmentos sin solapamiento que
  saldrían: 2851 con 2 s, 1889 con 3 s y 1118 con 5 s (Tug: 296, 196 y 118).
- **Alternativas:** segmentos de 2, 3 o 5 s; ventana de LOFAR de 0,5 s (2 Hz) o 0,25 s (4 Hz).
- **Decisión:** todos los valores viven como constantes en un único módulo.
  - **Preprocesado:** se resta el offset DC por grabación (D-009); segmentos de **3 s sin
    solapamiento** (se descarta la cola que no completa un segmento); cada segmento se escala a
    RMS = 1.
  - **LOFAR:** diezmado de 32 a 4 kHz con filtro antialiasing; banda de 0 a 2 kHz; ventana Hann
    de 0,5 s con avance de 0,25 s (2 Hz de resolución; 11 tramas × 1001 bins por segmento);
    normalización de fondo con TPSW (ventana de ±20 bins, hueco de ±2 bins, recorte en la
    segunda pasada de los valores mayores que 2 veces el fondo); salida 10·log10(potencia /
    fondo), en dB sobre el fondo.
  - **DEMON:** paso banda Butterworth de orden 4 de 2 a 8 kHz, en ambos sentidos; envolvente
    |Hilbert| menos su media; diezmado a 500 Hz en dos etapas de ×8; espectro del segmento con
    ventana Hann (0,33 Hz de resolución; 301 bins de 0 a 100 Hz); bins por debajo de 1 Hz a 0;
    normalizado por su máximo.
  - **Salida en `datasets/processed/`** (ignorada por git): un `.npz` por partición (`lofar`
    (N, 11, 1001) y `demon` (N, 301) en float32, más los ejes de frecuencia); `index.csv` con
    `segment_id`, `wav_path`, `vessel_class`, `vessel_group`, `split`, `segment_index` y
    `start_s`; y `features_meta.json` con los parámetros, el SHA-256 de `split.csv` y el commit
    del proyecto. Determinista, sin azar.
- **Motivo:** 3 s es habitual con DeepShip, cabe dos veces en la grabación más corta (D-009 no
  cambia) y deja 196 segmentos de Tug. Escalar por segmento quita el nivel absoluto, que depende
  de la distancia (atenúa DT-005). La ventana de 0,5 s prima la resolución en frecuencia, que es
  para lo que sirve LOFAR; en 3 s las líneas tonales son estables y el eje del tiempo aporta poco.
- **Ajuste (2026-10-11):** dentro de Docker no hay `.git` ni `git`, así que `project_commit` queda
  `null` en `features_meta.json`. El código se identifica con `code_sha256` (hash de los `.py` del
  paquete), junto al SHA-256 de `split.csv` y las versiones de numpy y scipy. Se descartó pasar el
  commit por variable de entorno o montar `.git` en el contenedor. El mismo criterio vale para
  `metrics.json` en F3. El espectro DEMON es de magnitud. Aprobado por David.

### D-016 · matplotlib para las figuras
- **Fecha:** 2026-10-11
- **Contexto:** el entregable de F2 incluye figuras de ejemplo (F2-T08), y F3 y F4 necesitarán
  matrices de confusión, curvas ROC y el informe. CLAUDE.md prohíbe un modo demo con datos
  sintéticos.
- **Alternativas:** (a) matplotlib; (b) no hacer figuras; (c) otra librería de gráficos.
- **Decisión:** (a), con la versión fijada en `requirements.txt` y el backend `Agg` (PNG sin
  pantalla). Las figuras de los datos reales se generan con `features --figures` en
  `outputs/figures/`; las de validación con señales sintéticas, desde un test con la opción
  `--figures-dir`, para que las señales sintéticas sigan existiendo solo en `tests/`.
- **Motivo:** es la librería estándar, se usará en F3 y F4, y añade unos 30-40 MB a la imagen.

## Deuda técnica

| ID | Descripción | Por qué se acepta | Impacto | Cuándo pagarla | Estado |
|---|---|---|---|---|---|
| DT-001 | El UID 1000 del usuario `app` está fijo en la imagen. | Es el UID habitual del primer usuario en Linux, y en Docker Desktop (Windows y macOS) no afecta. | En Linux, con otro UID, el contenedor no puede escribir en los volúmenes montados (`outputs/`). | En F4, si el friend-tester lo detecta. | Abierta |
| DT-002 | La asignación WAV → buque no viene dada directamente: 35 WAV se unen por id, 26 se deducen comparando su duración con la del metafile y 2 (Passengership 30 y 37) no cuadran por duración. | En los 61 casos verificados la duración coincide con un margen de 1 s, y los 2 dudosos se agrupan con COLUMBIA (D-007), así que no pueden separarse entre particiones. | Si una deducción fuera errónea, una grabación podría ir a la partición equivocada y provocar una fuga. | En F1-T06b: un test que compruebe que la duración de cada WAV coincide con la de su fila asignada en los 61 casos verificados. | Pagada (2026-10-11): `tests/test_vessel_table_data.py` |
| DT-003 | Algunos nombres de buque de los metafiles parecen cortados ("QUEEN OF", "CABO DE", "OTTO H", "SPIRIT OF"...). Un mismo buque podría aparecer con dos nombres distintos y formar dos grupos. | No se ha visto en los 44 grupos con WAV; los nombres cortados conocidos sin WAV no afectan a la partición, y "QUEEN OF" ya es un solo grupo (D-007). | Si ocurriera, el mismo buque podría acabar en dos particiones (fuga). | En F1-T06b: listar los grupos con WAV cuyos nombres se parezcan (p. ej., mismo prefijo) y revisarlos con David. | Pagada (2026-10-11): listado revisado con David; D-011 |
| DT-004 | Ningún test regenera `vessels.csv` desde los audios: si cambian las reglas de `build_vessel_table.py` (agrupación, alineamiento) y no se regenera la tabla, los tests siguen en verde con la tabla antigua. | La tabla actual está comprobada a mano y por los tests de DT-002. | El script y la tabla podrían divergir sin aviso; al regenerarla cambiarían grupos y partición sin que nadie lo haya decidido. | Al empezar F2: un test que ejecute el script sobre `datasets/raw/DeepShip`, escriba en una carpeta temporal y compare byte a byte con la versionada; se salta si no hay audios. | Pagada (2026-10-11): `tests/test_vessel_table_regen.py` |
| DT-005 | La clase coincide con la época de grabación: Cargo y Tug se grabaron en noviembre y diciembre de 2017, y Passengership y Tanker en 2016. Además, la duración depende de la clase (D-009). | No es una fuga entre particiones y no se puede corregir con estos datos. | Un modelo podría aprender la época, las condiciones del mar o la duración en vez de la firma del buque, y los resultados parecerían mejores de lo que son. | En F3: comprobar que un clasificador trivial (solo duración o ruido de fondo) no acierta demasiado, e informarlo junto a los resultados. | Abierta |
| DT-006 | Mejoras técnicas menores de la revisión de F1: la asignación por id toma la primera fila sin avisar si el id estuviera repetido (`build_vessel_table.py`); `qc.csv` usa CRLF y orden lexicográfico; un test de `qc` compara solo nombres y no contenido; el test de `--help` busca subcadenas; `python:3.11-slim` sin versión exacta y el índice de PyTorch aplicado a todas las dependencias; las extensiones de audio de `.gitignore` distinguen mayúsculas (el hook `forbid-audio` sí lo cubre); los tests de datos fallan en vez de saltarse si se usa `docker run` sin compose. | Ninguna afecta hoy a los datos ni a la partición. | Bajo: fragilidad y pequeñas incoherencias. | En F2, al tocar esos ficheros. | Pagada (2026-10-11): F2-T02 |
