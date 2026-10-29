"""Fondos de la presentación, imitando la propuesta que el grupo armó en Canva
(proyecto/1_propuesta/VotoGrafo_propuesta.pdf): degradado azul noche con una red tenue para portada y cierre, y gris
lavanda con racimos de red en dos esquinas para el contenido.

Uso: python fondos.py  (deja figuras/fondo_oscuro.png y figuras/fondo_claro.png)
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

FIG = Path(__file__).resolve().parent.parent / "figuras"
W, H = 1920, 1080


def degradado(c0, c1, c2):
    """c0 arriba a la izquierda, c1 arriba a la derecha, c2 abajo a la izquierda."""
    x = np.linspace(0, 1, W)[None, :, None]
    y = np.linspace(0, 1, H)[:, None, None]
    a, b, c = (np.array(to_rgb(k))[None, None, :] for k in (c0, c1, c2))
    return np.clip(a + (b - a) * x * (1 - y) + (c - a) * y * (1 - x), 0, 1)


def red(ax, rng, n, cx, cy, rx, ry, color, alpha, k=3):
    pts = np.column_stack([rng.normal(cx, rx, n), rng.normal(cy, ry, n)])
    for i, p in enumerate(pts):
        d = np.linalg.norm(pts - p, axis=1)
        for j in np.argsort(d)[1:k + 1]:
            ax.plot(*zip(p, pts[j]), color=color, alpha=alpha, lw=0.8)
    ax.scatter(pts[:, 0], pts[:, 1], s=rng.uniform(4, 22, n), color=color, alpha=min(1, alpha * 2.2), lw=0)


def lienzo(img):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, extent=(0, W, H, 0))
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    return fig, ax


def main():
    rng = np.random.default_rng(7)
    fig, ax = lienzo(degradado("#16223F", "#2A2348", "#123A52"))
    red(ax, rng, 110, W * 0.45, H * 0.45, W * 0.32, H * 0.3, "#C9D3E6", 0.13)
    fig.savefig(FIG / "fondo_oscuro.png"); plt.close(fig)

    fig, ax = lienzo(degradado("#F2F3F8", "#ECEEF6", "#EEF0F7"))
    red(ax, rng, 26, W * 0.80, H * 0.08, W * 0.06, H * 0.12, "#9CC0E4", 0.35)
    red(ax, rng, 22, W * 0.10, H * 0.92, W * 0.07, H * 0.07, "#9CC0E4", 0.35)
    fig.savefig(FIG / "fondo_claro.png"); plt.close(fig)
    print("fondos en", FIG)


if __name__ == "__main__":
    main()
