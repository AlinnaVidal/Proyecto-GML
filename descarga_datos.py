"""Descarga las votaciones de sala del período legislativo 2022-2026 desde la API
de datos abiertos de la Cámara de Diputadas y Diputados y las guarda como CSV
dentro del repositorio, en la carpeta ``datos/`` (en la raíz del repo).

Uso (desde cualquier carpeta del repo):
    python descarga_datos.py
    python descarga_datos.py --inicio 2024-01-01 --fin 2024-12-31
    python descarga_datos.py --salida datos --workers 4 --pausa 0.2
    python descarga_datos.py --refrescar      # ignora la caché

La primera ejecución tarda unos 20 minutos. Las respuestas XML quedan en
``datos/cache/`` (ignorada por git), así que una segunda ejecución es casi
instantánea y, si una descarga se corta, al reintentar se retoma donde quedó.

Archivos generados en ``datos/``:
    diputados.csv, militancias.csv, votaciones.csv, votos.csv,
    proyectos.csv, autores.csv, metadata.json, README.md, .gitignore
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://opendata.camara.cl/camaradiputados/WServices"
NS = {"c": "http://opendata.camara.cl/camaradiputados/v1"}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120 Safari/537.36"}
INICIO_DEFECTO, FIN_DEFECTO = "2022-03-11", "2026-03-11"

log = logging.getLogger("descarga")


# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
def raiz_repo() -> Path:
    """Sube desde este archivo hasta encontrar la carpeta .git.
    Si no hay repo git, usa la carpeta padre del script."""
    aqui = Path(__file__).resolve().parent
    for p in [aqui, *aqui.parents]:
        if (p / ".git").exists():
            return p
    return aqui.parent


# --------------------------------------------------------------------------- #
# Cliente HTTP con caché en disco
# --------------------------------------------------------------------------- #
class Cliente:
    """GET con caché en disco, reintentos y una sesión por hilo."""

    def __init__(self, cache: Path, refrescar: bool = False, pausa: float = 0.0):
        self.cache = cache
        self.refrescar = refrescar
        self.pausa = pausa
        self._local = threading.local()
        self.cache.mkdir(parents=True, exist_ok=True)

    def _sesion(self) -> requests.Session:
        if not hasattr(self._local, "s"):
            s = requests.Session()
            s.headers.update(UA)
            self._local.s = s
        return self._local.s

    def xml(self, servicio: str, metodo: str, **params) -> ET.Element:
        clave = f"{servicio}/{metodo}?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        ruta = self.cache / (hashlib.md5(clave.encode()).hexdigest() + ".xml")

        if ruta.exists() and not self.refrescar:
            try:
                return ET.fromstring(ruta.read_bytes())
            except ET.ParseError:       # archivo de caché corrupto: se vuelve a bajar
                ruta.unlink()

        url = f"{BASE}/{servicio}.asmx/{metodo}"
        ultimo = None
        for intento in range(5):
            try:
                r = self._sesion().get(url, params=params, timeout=60)
                r.raise_for_status()
                raiz = ET.fromstring(r.content)   # se valida antes de guardar en caché
                tmp = ruta.with_suffix(".tmp")
                tmp.write_bytes(r.content)
                tmp.replace(ruta)                 # escritura atómica
                if self.pausa:
                    time.sleep(self.pausa)
                return raiz
            except (requests.RequestException, ET.ParseError) as e:
                ultimo = e
                time.sleep(2 ** intento)
        raise RuntimeError(f"falló {url} {params}: {ultimo}")


def txt(nodo, ruta, attr=None):
    x = nodo.find(ruta, NS)
    if x is None:
        return None
    return x.get(attr) if attr else (x.text or "").strip()


# --------------------------------------------------------------------------- #
# Descargas
# --------------------------------------------------------------------------- #
def descargar_diputados(cli: Cliente):
    raiz = cli.xml("WSDiputado", "retornarDiputados")
    filas, milit = [], []
    for d in raiz.findall("c:Diputado", NS):
        did = int(txt(d, "c:Id"))
        filas.append({
            "diputado_id": did,
            "nombre": " ".join(filter(None, [txt(d, "c:Nombre"),
                                             txt(d, "c:ApellidoPaterno"),
                                             txt(d, "c:ApellidoMaterno")])),
            "sexo": txt(d, "c:Sexo"),
            "nacimiento": txt(d, "c:FechaNacimiento"),
        })
        for m in d.findall("c:Militancias/c:Militancia", NS):
            milit.append({
                "diputado_id": did,
                "inicio": txt(m, "c:FechaInicio"),
                "termino": txt(m, "c:FechaTermino"),
                "partido": txt(m, "c:Partido/c:Alias") or txt(m, "c:Partido/c:Id"),
                "partido_nombre": txt(m, "c:Partido/c:Nombre"),
            })
    return pd.DataFrame(filas), pd.DataFrame(milit)


def descargar_lista_votaciones(cli: Cliente, inicio: pd.Timestamp, fin: pd.Timestamp):
    filas = []
    for anno in range(inicio.year, fin.year + 1):
        raiz = cli.xml("WSLegislativo", "retornarVotacionesXAnno", prmAnno=anno)
        for v in raiz.findall("c:Votacion", NS):
            filas.append({
                "votacion_id": int(txt(v, "c:Id")),
                "descripcion": txt(v, "c:Descripcion"),
                "fecha": txt(v, "c:Fecha"),
                "total_si": int(txt(v, "c:TotalSi")),
                "total_no": int(txt(v, "c:TotalNo")),
                "total_abst": int(txt(v, "c:TotalAbstencion")),
                "total_disp": int(txt(v, "c:TotalDispensado")),
                "quorum": txt(v, "c:Quorum"),
                "resultado": txt(v, "c:Resultado"),
                "tipo": txt(v, "c:Tipo"),
            })
    df = pd.DataFrame(filas).drop_duplicates("votacion_id")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df[(df.fecha >= inicio) & (df.fecha < fin)].sort_values(["fecha", "votacion_id"])
    # Primer número de boletín que aparece en la descripción (ej. 14077-18).
    df["boletin"] = df.descripcion.str.extract(r"(\d+-\d+)")[0]
    return df.reset_index(drop=True)


def votos_de(cli: Cliente, vid: int):
    """Devuelve (votacion_id, lista de votos) o (votacion_id, None) si falla."""
    try:
        raiz = cli.xml("WSLegislativo", "retornarVotacionDetalle", prmVotacionId=vid)
        return vid, [{"votacion_id": vid,
                      "diputado_id": int(txt(v, "c:Diputado/c:Id")),
                      "opcion": txt(v, "c:OpcionVoto")}
                     for v in raiz.findall("c:Votos/c:Voto", NS)]
    except Exception as e:
        log.warning("votación %s: %s", vid, e)
        return vid, None


def proyecto_de(cli: Cliente, boletin: str):
    """Devuelve (boletin, proyecto, autores, detalle_votaciones); proyecto=None si no existe o falla."""
    try:
        raiz = cli.xml("WSLegislativo", "retornarProyectoLey", prmNumeroBoletin=boletin)
    except Exception as e:
        log.warning("boletín %s: %s", boletin, e)
        return boletin, "ERROR", [], []
    if raiz.find("c:NumeroBoletin", NS) is None:
        return boletin, None, [], []
    p = {
        "boletin": boletin,
        "nombre": txt(raiz, "c:Nombre"),
        "fecha_ingreso": txt(raiz, "c:FechaIngreso"),
        "iniciativa": txt(raiz, "c:TipoIniciativa"),
        "camara_origen": txt(raiz, "c:CamaraOrigen"),
        "ministerios": ";".join(m.text for m in raiz.findall(
            "c:MinisteriosPatrocinantes/c:Ministerio/c:Nombre", NS) if m.text),
        "materias": ";".join((m.text or "") for m in raiz.findall("c:Materias/c:Materia/c:Nombre", NS)),
    }
    autores = [{"boletin": boletin,
                "diputado_id": int(txt(a, "c:Diputado/c:Id")),
                "orden": txt(a, "c:Orden")}
               for a in raiz.findall("c:Autores/c:ParlamentarioAutor", NS)
               if txt(a, "c:Diputado/c:Id")]
    detalle = [{"votacion_id": int(txt(v, "c:Id")),
                "tipo_votacion": txt(v, "c:TipoVotacionProyectoLey"),
                "articulo": txt(v, "c:Articulo"),
                "tramite": txt(v, "c:TramiteConstitucional"),
                "tramite_reglamentario": txt(v, "c:TramiteReglamentario")}
               for v in raiz.findall("c:Votaciones/c:VotacionProyectoLey", NS)]
    return boletin, p, autores, detalle


# --------------------------------------------------------------------------- #
# Documentación de los datos
# --------------------------------------------------------------------------- #
README = """# Datos

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
"""


def escribir_documentacion(salida: Path, meta: dict):
    (salida / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (salida / "README.md").write_text(README, encoding="utf-8")
    (salida / ".gitignore").write_text("cache/\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inicio", default=INICIO_DEFECTO, help="fecha inicial (AAAA-MM-DD)")
    ap.add_argument("--fin", default=FIN_DEFECTO, help="fecha final, exclusiva (AAAA-MM-DD)")
    ap.add_argument("--salida", default=None,
                    help="carpeta de salida; por defecto <raíz del repo>/datos")
    ap.add_argument("--workers", type=int, default=6, help="descargas en paralelo")
    ap.add_argument("--pausa", type=float, default=0.0,
                    help="segundos de espera tras cada descarga nueva (para no saturar el servidor)")
    ap.add_argument("--refrescar", action="store_true", help="ignora la caché y vuelve a descargar")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    inicio, fin = pd.Timestamp(args.inicio), pd.Timestamp(args.fin)
    salida = Path(args.salida) if args.salida else raiz_repo() / "datos"
    salida.mkdir(parents=True, exist_ok=True)
    cli = Cliente(salida / "cache", refrescar=args.refrescar, pausa=args.pausa)
    log.info("salida: %s", salida)
    t0 = time.time()

    # 1. Diputados y militancias
    diputados, militancias = descargar_diputados(cli)
    diputados.sort_values("diputado_id").to_csv(salida / "diputados.csv", index=False)
    militancias.to_csv(salida / "militancias.csv", index=False)
    log.info("diputados: %d  militancias: %d", len(diputados), len(militancias))

    # 2. Lista de votaciones del período
    vot = descargar_lista_votaciones(cli, inicio, fin)
    log.info("votaciones en el período: %d", len(vot))

    # 3. Votos individuales
    with ThreadPoolExecutor(args.workers) as ex:
        resultados = list(ex.map(lambda v: votos_de(cli, v), vot.votacion_id))
    fallidas_votos = [vid for vid, lista in resultados if lista is None]
    votos = pd.DataFrame([f for _, lista in resultados if lista for f in lista])
    votos = votos.sort_values(["votacion_id", "diputado_id"])
    votos.to_csv(salida / "votos.csv", index=False)
    log.info("votos individuales: %d  (%.0f s)", len(votos), time.time() - t0)

    # 4. Proyectos, autores y detalle de cada votación
    boletines = sorted(vot.boletin.dropna().unique())
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(lambda b: proyecto_de(cli, b), boletines))
    fallidos_proy = [b for b, p, _, _ in res if p == "ERROR"]
    sin_proyecto = [b for b, p, _, _ in res if p is None]
    proyectos = pd.DataFrame([p for _, p, _, _ in res if isinstance(p, dict)])
    autores = pd.DataFrame([a for _, _, aa, _ in res for a in aa])
    detalle = pd.DataFrame([d for _, _, _, dd in res for d in dd])
    if not detalle.empty:
        detalle = detalle.drop_duplicates("votacion_id")
    proyectos.to_csv(salida / "proyectos.csv", index=False)
    autores.to_csv(salida / "autores.csv", index=False)

    vot = vot.merge(detalle, on="votacion_id", how="left") if not detalle.empty else vot
    vot.to_csv(salida / "votaciones.csv", index=False)
    log.info("proyectos: %d  autorías: %d  (%.0f s)", len(proyectos), len(autores), time.time() - t0)

    # 5. Chequeos básicos
    sin_votos = sorted(set(vot.votacion_id) - set(votos.votacion_id))
    log.info("opciones de voto: %s", votos.opcion.value_counts().to_dict())
    if sin_votos:
        log.warning("%d votaciones sin votos individuales", len(sin_votos))
    if fallidas_votos or fallidos_proy:
        log.warning("descargas fallidas: %d votaciones, %d proyectos. Vuelve a ejecutar el "
                    "script: lo ya descargado está en caché y solo se reintenta lo que falló.",
                    len(fallidas_votos), len(fallidos_proy))

    escribir_documentacion(salida, {
        "fuente": "https://camara.cl/transparencia/datosAbiertos.aspx",
        "api": BASE,
        "fecha_descarga_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "periodo": {"inicio": str(inicio.date()), "fin_exclusivo": str(fin.date())},
        "conteos": {
            "diputados": len(diputados), "militancias": len(militancias),
            "votaciones": len(vot), "votos": len(votos),
            "proyectos": len(proyectos), "autorias": len(autores),
        },
        "votaciones_sin_votos": len(sin_votos),
        "votaciones_sin_boletin": int(vot.boletin.isna().sum()),
        "boletines_sin_proyecto": len(sin_proyecto),
        "descargas_fallidas": {"votaciones": fallidas_votos, "proyectos": fallidos_proy},
    })
    log.info("listo en %.0f s", time.time() - t0)


if __name__ == "__main__":
    main()