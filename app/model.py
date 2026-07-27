"""Capa de dominio: carga del modelo entrenado y clasificación de muestras.

Este módulo no sabe nada de Tkinter. Solo recibe cuatro medidas numéricas y
devuelve una predicción. Gracias a eso puede reutilizarse desde la interfaz
gráfica, desde un script de consola o desde una API web sin modificar nada.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

# Orden de las características tal como se entrenó el modelo en la Actividad 1.
# Es el orden estándar del iris dataset de scikit-learn y de Keras.
# Si tu entrenamiento usó otro orden, cámbialo AQUÍ: la interfaz lee esta lista.
FEATURE_NAMES: tuple[str, ...] = (
    "Largo del sépalo (cm)",
    "Ancho del sépalo (cm)",
    "Largo del pétalo (cm)",
    "Ancho del pétalo (cm)",
)

# Orden de las clases en la capa de salida (índice 0, 1, 2).
CLASS_NAMES: tuple[str, ...] = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")

# Rango admitido para las medidas, en centímetros.
MIN_VALUE = 0.0
MAX_VALUE = 30.0


class ModelError(Exception):
    """Error controlado de la capa de modelo (archivo ausente, salida inválida...)."""


@dataclass(frozen=True)
class Prediction:
    """Resultado de una clasificación. Es un dato inmutable, sin lógica pesada."""

    label: str
    confidence: float
    probabilities: Mapping[str, float]

    def as_text(self) -> str:
        return f"{self.label}  (confianza: {self.confidence:.1%})"


class Classifier(Protocol):
    """Contrato mínimo que la interfaz necesita.

    La ventana depende de este protocolo, no de IrisClassifier. Eso permite
    probar la interfaz con un clasificador falso sin cargar TensorFlow.
    """

    def predict(self, features: Sequence[float]) -> Prediction: ...


def validate_features(features: Sequence[float]) -> tuple[float, ...]:
    """Comprueba cantidad, tipo y rango de las medidas. Lanza ValueError si algo falla."""
    try:
        values = tuple(float(value) for value in features)
    except (TypeError, ValueError) as exc:
        raise ValueError("Todas las medidas deben ser números.") from exc

    if len(values) != len(FEATURE_NAMES):
        raise ValueError(
            f"Se esperaban {len(FEATURE_NAMES)} medidas y se recibieron {len(values)}."
        )

    for name, value in zip(FEATURE_NAMES, values):
        if not math.isfinite(value):
            raise ValueError(f"«{name}» no es un número válido.")
        if not MIN_VALUE < value <= MAX_VALUE:
            raise ValueError(
                f"«{name}» debe estar entre {MIN_VALUE} y {MAX_VALUE} cm. "
                f"Valor recibido: {value}"
            )
    return values


def _softmax(vector: np.ndarray) -> np.ndarray:
    """Convierte logits en probabilidades. Se resta el máximo por estabilidad numérica."""
    exponentials = np.exp(vector - np.max(vector))
    return exponentials / np.sum(exponentials)


class IrisClassifier:
    """Envuelve el modelo Keras (.h5) y expone una única operación: predict()."""

    def __init__(
        self,
        model_path: str | Path,
        class_names: Sequence[str] = CLASS_NAMES,
        preprocess: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """
        model_path : ruta del archivo .h5 generado en la Actividad 1.
        class_names: nombres de las clases, en el mismo orden que la capa de salida.
        preprocess : función opcional para escalar la entrada. Se usa solo si en
                     la Actividad 1 entrenaste con StandardScaler o MinMaxScaler.
        """
        self._model_path = Path(model_path)
        self._class_names = tuple(class_names)
        self._preprocess = preprocess
        self._model = None

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Carga el modelo en memoria. Se llama una sola vez, al iniciar la app."""
        if self._model is not None:
            return

        if not self._model_path.is_file():
            raise ModelError(
                f"No se encontró el archivo del modelo en:\n{self._model_path}\n\n"
                "Copia 'modelo_iris.h5' (Actividad 1) dentro de la carpeta 'models/'."
            )

        try:
            from tensorflow import keras  # import diferido: acelera el arranque
        except ImportError as exc:
            raise ModelError(
                "TensorFlow no está instalado.\n"
                "Ejecuta:  pip install -r requirements.txt"
            ) from exc

        try:
            # compile=False: para inferencia no se necesita el optimizador y
            # evita advertencias al abrir modelos .h5 guardados con otra versión.
            self._model = keras.models.load_model(self._model_path, compile=False)
        except Exception as exc:  # noqa: BLE001 - se traduce a un error del dominio
            raise ModelError(f"No se pudo cargar el modelo:\n{exc}") from exc

    def predict(self, features: Sequence[float]) -> Prediction:
        """Clasifica una muestra. Lanza ValueError si los datos son inválidos."""
        if self._model is None:
            self.load()

        values = validate_features(features)
        sample = np.asarray([values], dtype="float32")

        if self._preprocess is not None:
            sample = self._preprocess(sample)

        try:
            raw_output = self._model.predict(sample, verbose=0)
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"Fallo al ejecutar la predicción:\n{exc}") from exc

        probabilities = self._to_probabilities(raw_output)
        best = int(np.argmax(probabilities))

        return Prediction(
            label=self._class_names[best],
            confidence=float(probabilities[best]),
            probabilities={
                name: float(value)
                for name, value in zip(self._class_names, probabilities)
            },
        )

    def _to_probabilities(self, raw_output) -> np.ndarray:
        """Normaliza la salida del modelo a probabilidades que suman 1."""
        vector = np.asarray(raw_output, dtype="float64").ravel()

        if vector.size != len(self._class_names):
            raise ModelError(
                f"El modelo devolvió {vector.size} salidas y se esperaban "
                f"{len(self._class_names)}. Revisa la última capa de la red."
            )

        # Si la última capa no es softmax, la salida son logits: se normaliza aquí.
        if vector.min() < 0 or not math.isclose(float(vector.sum()), 1.0, abs_tol=1e-3):
            vector = _softmax(vector)

        return vector
