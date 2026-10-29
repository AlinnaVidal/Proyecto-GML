"""Descarga las votaciones de sala del período legislativo 2022-2026 desde la API
de datos abiertos de la Cámara de Diputados y las guarda como CSV en ../datos/.

Uso: python descarga_datos.py
Tarda unos 20 minutos la primera vez. Las respuestas XML quedan en ../datos/cache/
así que una segunda ejecución es casi instantánea.
"""
import re
import time
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

BASE = "https://opendata.camara.cl/camaradiputados/WServices"
NS = {"c": "http://opendata.camara.cl/camaradiputados/v1"}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120 Safari/537.36"}
INICIO, FIN = pd.Timestamp("2022-03-11"), pd.Timestamp("2026-03-11")

RAIZ = Path(__file__).resolve().parent.parent / "datos"
CACHE = RAIZ / "cache"


def get_xml(servicio, metodo, **params):
    """GET con caché en disco y reintentos."""
    clave = metodo + "_" + "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    ruta = CACHE / (hashlib.md5(clave.encode()).hexdigest() + ".xml")
    if ruta.exists():
        return ET.fromstring(ruta.read_bytes())
    url = f"{BASE}/{servicio}.asmx/{metodo}"
    for intento in range(5):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=60)
            r.raise_for_status()
            ruta.write_bytes(r.content)
            return ET.fromstring(r.content)
        except Exception:
            time.sleep(2 ** intento)
    raise RuntimeError(f"falló {url} {params}")


def txt(nodo, ruta, attr=None):
    x = nodo.find(ruta, NS)
    if x is None:
        return None
    return x.get(attr) if attr else (x.text or "").strip()


def descargar_diputados():
    raiz = get_xml("WSDiputado", "retornarDiputados")
    filas, milit = [], []
    for d in raiz.findall("c:Diputado", NS):
        did = int(txt(d, "c:Id"))
        filas.append({
            "diputado_id": did,
            "nombre": f'{txt(d, "c:Nombre")} {txt(d, "c:ApellidoPaterno")} {txt(d, "c:ApellidoMaterno")}',
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


def descargar_lista_votaciones():
    filas = []
    for anno in range(INICIO.year, FIN.year + 1):
        raiz = get_xml("WSLegislativo", "retornarVotacionesXAnno", prmAnno=anno)
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
    df = df[(df.fecha >= INICIO) & (df.fecha < FIN)].sort_values("fecha")
    df["boletin"] = df.descripcion.str.extract(r"(\d+-\d+)")[0]
    return df.reset_index(drop=True)


def votos_de(vid):
    raiz = get_xml("WSLegislativo", "retornarVotacionDetalle", prmVotacionId=vid)
    return [{"votacion_id": vid,
             "diputado_id": int(txt(v, "c:Diputado/c:Id")),
             "opcion": txt(v, "c:OpcionVoto")}
            for v in raiz.findall("c:Votos/c:Voto", NS)]


def proyecto_de(boletin):
    raiz = get_xml("WSLegislativo", "retornarProyectoLey", prmNumeroBoletin=boletin)
    if raiz.find("c:NumeroBoletin", NS) is None:
        return None, [], []
    p = {
        "boletin": boletin,
        "nombre": txt(raiz, "c:Nombre"),
        "fecha_ingreso": txt(raiz, "c:FechaIngreso"),
        "iniciativa": txt(raiz, "c:TipoIniciativa"),
        "camara_origen": txt(raiz, "c:CamaraOrigen"),
        "ministerios": ";".join(m.text for m in raiz.findall(
            "c:MinisteriosPatrocinantes/c:Ministerio/c:Nombre", NS)),
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
    return p, autores, detalle


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    diputados, militancias = descargar_diputados()
    diputados.to_csv(RAIZ / "diputados.csv", index=False)
    militancias.to_csv(RAIZ / "militancias.csv", index=False)
    print(f"diputados: {len(diputados)}  militancias: {len(militancias)}")

    vot = descargar_lista_votaciones()
    print(f"votaciones en el período: {len(vot)}")

    with ThreadPoolExecutor(6) as ex:
        votos = [f for lista in ex.map(votos_de, vot.votacion_id) for f in lista]
    votos = pd.DataFrame(votos)
    votos.to_csv(RAIZ / "votos.csv", index=False)
    print(f"votos individuales: {len(votos)}  ({time.time() - t0:.0f} s)")

    boletines = sorted(vot.boletin.dropna().unique())
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(proyecto_de, boletines))
    proyectos = pd.DataFrame([p for p, _, _ in res if p])
    autores = pd.DataFrame([a for _, aa, _ in res for a in aa])
    detalle = pd.DataFrame([d for _, _, dd in res for d in dd]).drop_duplicates("votacion_id")
    proyectos.to_csv(RAIZ / "proyectos.csv", index=False)
    autores.to_csv(RAIZ / "autores.csv", index=False)

    vot = vot.merge(detalle, on="votacion_id", how="left")
    vot.to_csv(RAIZ / "votaciones.csv", index=False)
    print(f"proyectos: {len(proyectos)}  autorías: {len(autores)}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
