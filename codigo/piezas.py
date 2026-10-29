"""Piezas de dibujo que usan las slides de avance.

- figuras/esquema_grafo.png: esquema del grafo heterogéneo.
- figuras/hemiciclo.png: hemiciclo de la portada.

Uso: python piezas.py (antes, python fondos.py)
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

RAIZ = Path(__file__).resolve().parent.parent

TINTA, GRIS, GRIS_CLARO = "#13203A", "#5A6578", "#E8EBEF"
TURQUESA, AMBAR = "#0F7C83", "#E0A100"
SI, NO, ABS, ROJO = "#2E7D4F", "#C8102E", "#E0A100", "#C8102E"
TIPOS = {  # color de cada tipo de nodo
    "Diputado": "#1B2A41", "Votación": "#0F7C8C", "Proyecto": "#6A4C93",
    "Partido": "#1982C4", "Comisión": "#4A5A78",
}


def asientos(n=155, filas=8, r0=0.45, r1=1.0):
    radios = np.linspace(r0, r1, filas)
    cupos = np.round(n * radios / radios.sum()).astype(int)
    cupos[-1] += n - cupos.sum()
    pts = []
    for r, c in zip(radios, cupos):
        ang = np.linspace(np.pi, 0, c)
        pts += [(r * np.cos(a), r * np.sin(a), a) for a in ang]
    pts.sort(key=lambda p: -p[2])
    return np.array([(x, y) for x, y, _ in pts])


def hemiciclo(ax, cx, cy, escala, colores, s=16, marcas=None):
    p = asientos(len(colores)) * escala + np.array([cx, cy])
    ax.scatter(p[:, 0], p[:, 1], s=s, c=colores, edgecolors="white", linewidths=0.3, zorder=3)
    if marcas is not None:
        for i in marcas:
            ax.text(p[i, 0], p[i, 1], "?", ha="center", va="center", fontsize=s / 3.2, color="white",
                    fontweight="bold", zorder=4)


def esquema_grafo(ax, ox, oy, e=1.0):
    pos = {"Diputado": (0.0, 1.1), "Votación": (2.2, 1.1), "Proyecto": (3.2, -0.6), "Partido": (-0.6, -0.6),
           "Comisión": (4.6, -0.6), "Votación2": (2.2, 2.75)}
    pos = {k: (ox + x * e, oy + y * e) for k, (x, y) in pos.items()}
    aristas = [("Diputado", "Votación", "sí / no / abst.", TINTA, 0.0, (1.1, 1.32)),
               ("Diputado", "Partido", "milita", GRIS, 0.0, (-0.75, 0.45)),
               ("Diputado", "Proyecto", "autor", GRIS, 0.3, (1.15, -0.15)),
               ("Votación", "Proyecto", "de", GRIS, 0.0, (2.95, 0.5)),
               ("Proyecto", "Comisión", "en", GRIS, 0.0, (3.9, -0.1)),
               ("Votación2", "Votación", "similar / mismo proyecto", ABS, 0.0, (3.35, 1.85))]
    for a, b, et, col, rad, (lx, ly) in aristas:
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-", connectionstyle=f"arc3,rad={rad}",
                                     color=col, lw=2.6 if col != GRIS else 1.6, zorder=1,
                                     linestyle="--" if col == ABS else "-"))
        ax.text(ox + lx * e, oy + ly * e, et, fontsize=10.5, color=col, ha="center", va="center",
                style="italic", zorder=5)
    for k, (x, y) in pos.items():
        nombre = "Votación" if k.startswith("Votación") else k
        ax.add_patch(Circle((x, y), 0.36 * e, fc=TIPOS[nombre], ec="white", lw=2, zorder=3,
                            alpha=0.55 if k == "Votación2" else 1))
        etiqueta = "votación\npasada" if k == "Votación2" else nombre
        arriba = k in ("Votación2", "Diputado", "Votación")
        ax.text(x, y + 0.5 * e if arriba else y - 0.5 * e, etiqueta, ha="center",
                va="bottom" if arriba else "top", fontsize=11, color=TINTA, fontweight="bold", zorder=5)


def piezas():
    """Esquema del grafo y hemiciclo de portada como imágenes sueltas para las presentaciones."""
    fig = plt.figure(figsize=(7, 5.2)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1.6, 5.6); ax.set_ylim(-1.9, 3.6); ax.axis("off")
    esquema_grafo(ax, 0, 0, e=1.0)
    fig.savefig(RAIZ / "figuras" / "esquema_grafo.png", dpi=200, transparent=True); plt.close(fig)

    fig = plt.figure(figsize=(7, 3.8)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.05, 1.08); ax.axis("off")
    rng = np.random.default_rng(1)
    col = np.array(sorted(rng.choice([SI, NO, ABS], 155, p=[0.66, 0.27, 0.07]), key=lambda c: [SI, ABS, NO].index(c)))
    hemiciclo(ax, 0, 0, 1.0, col, s=95)
    fig.savefig(RAIZ / "figuras" / "hemiciclo.png", dpi=200, transparent=True); plt.close(fig)
    print("escrito", RAIZ / "figuras" / "esquema_grafo.png", "y hemiciclo.png")


if __name__ == "__main__":
    piezas()
