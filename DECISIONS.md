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

## Deuda técnica

| ID | Descripción | Por qué se acepta | Impacto | Cuándo pagarla | Estado |
|---|---|---|---|---|---|
| DT-001 | El UID 1000 del usuario `app` está fijo en la imagen. | Es el UID habitual del primer usuario en Linux, y en Docker Desktop (Windows y macOS) no afecta. | En Linux, con otro UID, el contenedor no puede escribir en los volúmenes montados (`outputs/`). | En F4, si el friend-tester lo detecta. | Abierta |
| DT-002 | La asignación WAV → buque no viene dada directamente: 35 WAV se unen por id, 26 se deducen comparando su duración con la del metafile y 2 (Passengership 30 y 37) no cuadran por duración. | En los 61 casos verificados la duración coincide con un margen de 1 s, y los 2 dudosos se agrupan con COLUMBIA (D-007), así que no pueden separarse entre particiones. | Si una deducción fuera errónea, una grabación podría ir a la partición equivocada y provocar una fuga. | En F1-T06/T07: un test que compruebe que la duración de cada WAV coincide con la de su fila asignada en los 61 casos verificados. | Abierta |
