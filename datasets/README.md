# Datos

## DeepShip

- **Contenido:** grabaciones de ruido radiado por buques, en 4 clases: Cargo, Passengership,
  Tanker y Tug.
- **Fuente:** https://github.com/irfankamboh/DeepShip, último commit
  `77725726a33d6aef4ce1d0fe70f67c0bbee4b5b6` (2023-03-23), revisado el 2026-10-10.
  - Es un **subconjunto** del dataset: 67 WAV, unos 733 MB en total (Cargo 13,
    Passengership 21, Tanker 29, Tug 4).
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
  `raw/DeepShip` y la fijan en el commit indicado arriba ([D-006](../DECISIONS.md)). Los comandos
  se añaden en la tarea F1-T04.
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

## Partición

La partición train/val/test se hace **por buque** si los metadatos de DeepShip lo permiten, y
si no, **por grabación**. Nunca por segmento, para evitar fugas de datos. El manifiesto con la
partición se versiona en `manifests/`.
