# Datos

Datos abiertos de la Cámara de Diputadas y Diputados de Chile
(https://camara.cl/transparencia/datosAbiertos.aspx), descargados con
`scripts/descarga_datos.py`. Ver `metadata.json` para fecha de descarga,
período y conteos.

| Archivo | Contenido |
|---|---|
| `diputados.csv` | un diputado por fila (id, nombre, sexo, nacimiento) |
| `militancias.csv` | partido de cada diputado con fecha de inicio y término |
| `votaciones.csv` | una votación de sala por fila (fecha, totales, quórum, resultado, boletín, tipo de votación) |
| `votos.csv` | un voto individual por fila (`votacion_id`, `diputado_id`, `opcion`) |
| `proyectos.csv` | un proyecto de ley por fila (boletín, nombre, iniciativa, materias) |
| `autores.csv` | autoría de proyectos (`boletin`, `diputado_id`, `orden`) |

`cache/` guarda las respuestas XML originales y no se sube a git.
Para regenerar todo: `python scripts/descarga_datos.py`.
