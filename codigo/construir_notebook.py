"""Convierte notebook_src.py (celdas separadas por '# %%') en un .ipynb y, con
--ejecutar, lo corre completo con nbclient dejando las salidas guardadas.

Uso: python construir_notebook.py [--ejecutar]
"""
import sys
from pathlib import Path

import nbformat

AQUI = Path(__file__).resolve().parent
SRC = AQUI / "notebook_src.py"
DEST = AQUI.parent / "notebook" / "votaciones_gnn.ipynb"


def celdas(texto):
    bloques, actual, tipo = [], [], None
    for linea in texto.splitlines():
        if linea.startswith("# %%"):
            if tipo is not None:
                bloques.append((tipo, actual))
            tipo = "md" if "[markdown]" in linea else "code"
            actual = []
        elif tipo is not None:
            actual.append(linea)
    if tipo is not None:
        bloques.append((tipo, actual))

    nb_celdas = []
    for tipo, lineas in bloques:
        while lineas and not lineas[-1].strip():
            lineas.pop()
        while lineas and not lineas[0].strip():
            lineas.pop(0)
        if tipo == "md":
            fuente = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in lineas)
            nb_celdas.append(nbformat.v4.new_markdown_cell(fuente))
        else:
            nb_celdas.append(nbformat.v4.new_code_cell("\n".join(lineas)))
    return nb_celdas


def main():
    nb = nbformat.v4.new_notebook()
    nb.cells = celdas(SRC.read_text(encoding="utf-8"))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    DEST.parent.mkdir(exist_ok=True)
    if "--ejecutar" in sys.argv:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=-1, kernel_name="python3",
                       resources={"metadata": {"path": str(DEST.parent)}}).execute()
    nbformat.write(nb, DEST)
    print("escrito", DEST)


if __name__ == "__main__":
    main()
