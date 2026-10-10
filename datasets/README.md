# Datos

## DeepShip

- **Contenido:** grabaciones de ruido radiado por buques, en 4 clases: Cargo, Passengership,
  Tanker y Tug.
- **Fuente oficial:** PENDIENTE DE VERIFICAR (tarea F1-T01 en [PLAN.md](../PLAN.md)).
- **Licencia:** PENDIENTE DE VERIFICAR. Hasta confirmarla, no se redistribuye nada.
- **Cómo obtenerlo:** se decide en F1 (tarea F1-T02). La decisión quedará en
  [DECISIONS.md](../DECISIONS.md).

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
