"""Capa de presentación: ventana Tkinter.

Este módulo no importa TensorFlow ni sabe cómo se calcula una predicción.
Recibe ya construido un objeto que cumpla el protocolo Classifier, le pasa
las medidas y muestra el resultado. Esa inyección de dependencias es lo que
mantiene baja la dependencia entre la interfaz y el modelo.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from app.model import FEATURE_NAMES, Classifier, ModelError

# Muestra de ejemplo (una Iris-setosa típica) para el botón "Ejemplo".
SAMPLE_VALUES = ("5.1", "3.5", "1.4", "0.2")

PLACEHOLDER = "—"


class IrisWindow(tk.Tk):
    """Ventana principal del clasificador."""

    def __init__(self, classifier: Classifier) -> None:
        super().__init__()
        self._classifier = classifier
        self._inputs: list[tk.StringVar] = [tk.StringVar() for _ in FEATURE_NAMES]
        self._result = tk.StringVar(value=PLACEHOLDER)
        self._detail = tk.StringVar(value="")
        self._build_layout()

    # ------------------------------------------------------------------ vista

    def _build_layout(self) -> None:
        self.title("Clasificador Iris - Modelo Neuronal")
        self.resizable(False, False)

        container = ttk.Frame(self, padding=16)
        container.grid(row=0, column=0, sticky="nsew")

        header = ttk.Label(
            container,
            text="CLASIFICADOR IRIS - Modelo Neuronal",
            font=("TkDefaultFont", 12, "bold"),
        )
        header.grid(row=0, column=0, columnspan=2, pady=(0, 12))

        first_entry: ttk.Entry | None = None
        for index, name in enumerate(FEATURE_NAMES, start=1):
            ttk.Label(container, text=f"{name}:").grid(
                row=index, column=0, sticky="w", pady=3, padx=(0, 10)
            )
            entry = ttk.Entry(container, textvariable=self._inputs[index - 1], width=12)
            entry.grid(row=index, column=1, sticky="e", pady=3)
            entry.bind("<Return>", self._on_classify)
            if first_entry is None:
                first_entry = entry

        buttons = ttk.Frame(container)
        buttons.grid(row=len(FEATURE_NAMES) + 1, column=0, columnspan=2, pady=14)
        ttk.Button(buttons, text="CLASIFICAR", command=self._on_classify).grid(
            row=0, column=0, padx=4
        )
        ttk.Button(buttons, text="Ejemplo", command=self._on_sample).grid(
            row=0, column=1, padx=4
        )
        ttk.Button(buttons, text="Limpiar", command=self._on_clear).grid(
            row=0, column=2, padx=4
        )

        ttk.Separator(container, orient="horizontal").grid(
            row=len(FEATURE_NAMES) + 2, column=0, columnspan=2, sticky="ew"
        )

        ttk.Label(container, text="Resultado:").grid(
            row=len(FEATURE_NAMES) + 3, column=0, sticky="w", pady=(12, 0)
        )
        ttk.Label(
            container,
            textvariable=self._result,
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=len(FEATURE_NAMES) + 3, column=1, sticky="e", pady=(12, 0))

        ttk.Label(
            container,
            textvariable=self._detail,
            foreground="gray30",
            font=("TkDefaultFont", 9),
        ).grid(row=len(FEATURE_NAMES) + 4, column=0, columnspan=2, pady=(6, 0))

        if first_entry is not None:
            first_entry.focus_set()

    # ----------------------------------------------------------- controlador

    def _read_inputs(self) -> list[float]:
        """Convierte el texto de las cajas a números. Acepta coma o punto decimal."""
        values: list[float] = []
        for name, variable in zip(FEATURE_NAMES, self._inputs):
            text = variable.get().strip().replace(",", ".")
            if not text:
                raise ValueError(f"El campo «{name}» está vacío.")
            try:
                values.append(float(text))
            except ValueError:
                raise ValueError(
                    f"«{name}» debe ser un número. Valor escrito: {text}"
                ) from None
        return values

    def _on_classify(self, event: tk.Event | None = None) -> None:
        try:
            features = self._read_inputs()
            prediction = self._classifier.predict(features)
        except ValueError as exc:
            messagebox.showwarning("Datos inválidos", str(exc), parent=self)
            return
        except ModelError as exc:
            messagebox.showerror("Error del modelo", str(exc), parent=self)
            return

        self._result.set(prediction.as_text())
        self._detail.set(
            "   ".join(
                f"{name}: {value:.1%}"
                for name, value in prediction.probabilities.items()
            )
        )

    def _on_sample(self) -> None:
        for variable, value in zip(self._inputs, SAMPLE_VALUES):
            variable.set(value)

    def _on_clear(self) -> None:
        for variable in self._inputs:
            variable.set("")
        self._result.set(PLACEHOLDER)
        self._detail.set("")
