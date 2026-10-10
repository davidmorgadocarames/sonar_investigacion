# sonar_investigacion

Pipeline de procesado de señal sonar pasivo sobre el dataset DeepShip (4 clases: Cargo,
Passengership, Tanker, Tug):

```
WAV → control de calidad → preprocesado → LOFAR y DEMON → clasificador → evaluación → informe
```

> **Estado:** esqueleto. Este README se completa en la fase F4. Por ahora solo se puede
> construir la imagen y ver la ayuda.

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

2. Construye la imagen. La primera vez tarda unos minutos:

   ```bash
   docker compose build
   ```

3. Comprueba que funciona:

   ```bash
   docker compose run --rm pipeline --help
   ```

   Debe mostrar los comandos `qc`, `features`, `train`, `evaluate` y `report`. Todavía no
   están implementados.

## Datos

*Pendiente (fase F1).* Aquí se explicará cómo obtener el conjunto de test de DeepShip. Mientras
tanto, consulta [datasets/README.md](datasets/README.md).

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
