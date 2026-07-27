"""Punto de entrada de la aplicación.

Este archivo es el único que conoce a la vez el modelo y la interfaz: crea el
clasificador, lo carga y se lo entrega a la ventana. Se le llama "raíz de
composición" y es lo que permite que app/model.py y app/ui.py no se conozcan
entre sí más allá del contrato Classifier.
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from app.model import IrisClassifier, ModelError
from app.ui import IrisWindow

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "modelo_iris.h5"


def main() -> int:
    """Arranca la aplicación. Devuelve 0 si todo salió bien, 1 si hubo error."""
    classifier = IrisClassifier(MODEL_PATH)

    try:
        classifier.load()
    except ModelError as exc:
        _show_startup_error(str(exc))
        return 1

    window = IrisWindow(classifier)
    window.mainloop()
    return 0


def _show_startup_error(message: str) -> None:
    """Muestra el error en una ventana y también en consola, por si no hay entorno gráfico."""
    print(f"[ERROR] {message}", file=sys.stderr)
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("No se pudo iniciar la aplicación", message)
        root.destroy()
    except tk.TclError:
        pass  # sin entorno gráfico disponible: basta con el mensaje en consola


if __name__ == "__main__":
    raise SystemExit(main())
