# Clasificador Iris con Tkinter

Aplicación de escritorio que recibe las cuatro medidas de una flor de iris
(largo y ancho de sépalo y pétalo) y muestra la especie predicha por la red
neuronal entrenada en la Actividad 1, junto con el nivel de confianza.

Se ejecuta de forma local, sin conexión a internet ni servidor.

## Estructura del proyecto

    iris_app/
    ├── app/
    │   ├── __init__.py       Marca la carpeta como paquete de Python
    │   ├── model.py          Carga el .h5 y realiza las predicciones
    │   └── ui.py             Interfaz gráfica con Tkinter
    ├── models/
    │   └── modelo_iris.h5    Modelo entrenado en la Actividad 1
    ├── main.py               Punto de entrada
    ├── requirements.txt      Dependencias
    └── README.md             Este documento

## Arquitectura

La aplicación está dividida en tres capas. El flujo de datos es lineal y las
flechas indican quién depende de quién:

    main.py  (raíz de composición: construye y conecta)
       │
       ├──►  app/ui.py     (presentación: ventana, cajas de texto, botones)
       │        │
       │        └──► depende solo del contrato `Classifier`
       │
       └──►  app/model.py  (dominio: validación, carga del .h5, predicción)
                │
                └──►  models/modelo_iris.h5

Decisiones de diseño y por qué:

- **Alta cohesión.** Cada módulo tiene una sola responsabilidad. `model.py` se
  ocupa únicamente de convertir cuatro números en una predicción; `ui.py`
  únicamente de mostrar y recoger datos en pantalla.

- **Baja dependencia (bajo acoplamiento).** `ui.py` no importa TensorFlow y
  `model.py` no importa Tkinter. La ventana recibe el clasificador ya
  construido por parámetro (inyección de dependencias), en lugar de crearlo
  ella misma. Consecuencia práctica: se puede probar la interfaz pasándole un
  clasificador falso, y se puede reutilizar `model.py` desde una API web sin
  tocar una sola línea.

- **Dependencia hacia una abstracción.** `ui.py` está tipada contra el
  protocolo `Classifier`, no contra la clase `IrisClassifier`. Cambiar el
  modelo por otro (scikit-learn, PyTorch) solo exige respetar ese contrato.

- **Errores traducidos.** Cualquier fallo de TensorFlow se convierte en un
  `ModelError` con un mensaje entendible. La interfaz no maneja excepciones de
  librerías externas, solo `ValueError` (dato inválido) y `ModelError`.

- **Datos inmutables.** El resultado viaja en un `dataclass` congelado
  (`Prediction`), así ninguna capa puede modificar la salida de otra.

## Requisitos

- Python 3.9 o superior
- Tkinter (incluido con Python; en Linux: `sudo apt install python3-tk`)

## Instalación

    cd iris_app
    python -m venv .venv

    # Windows
    .venv\Scripts\activate
    # Linux / macOS
    source .venv/bin/activate

    pip install -r requirements.txt

## Antes de ejecutar

Copia el archivo `modelo_iris.h5` de la Actividad 1 dentro de la carpeta
`models/`. La aplicación lo busca exactamente en `models/modelo_iris.h5`; si no
lo encuentra, muestra un mensaje de error al iniciar y no abre la ventana.

## Ejecución

    python main.py

Se abre la ventana, se escriben las cuatro medidas en centímetros y se pulsa
CLASIFICAR (o Enter). El botón "Ejemplo" carga una muestra de prueba.

## Tres cosas que debes verificar con tu modelo de la Actividad 1

Son la causa más común de que la app funcione pero prediga mal.

1. **Orden de las características.** El código usa el orden estándar del iris
   dataset: largo sépalo, ancho sépalo, largo pétalo, ancho pétalo. Si
   entrenaste en otro orden, edita `FEATURE_NAMES` en `app/model.py`; la
   interfaz genera las cajas de texto a partir de esa lista, así que se ajusta
   sola.

2. **Escalado.** Si en la Actividad 1 aplicaste `StandardScaler` o
   `MinMaxScaler` a los datos de entrenamiento, tienes que aplicar el mismo
   escalador aquí. `IrisClassifier` acepta una función `preprocess` para eso.
   En `main.py`:

       import joblib
       scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")
       classifier = IrisClassifier(MODEL_PATH, preprocess=scaler.transform)

   Si entrenaste con los datos sin escalar, déjalo como está.

3. **Orden de las clases.** `CLASS_NAMES` asume setosa, versicolor, virginica
   (el orden que producen `load_iris()` y `to_categorical`). Si usaste otro,
   corrígelo en `app/model.py`.

## Prueba rápida sin abrir la interfaz

Útil para comprobar que el modelo carga bien antes de depurar la ventana:

    python -c "from app.model import IrisClassifier; c = IrisClassifier('models/modelo_iris.h5'); print(c.predict([5.1, 3.5, 1.4, 0.2]))"

Con una setosa clara como esa, se espera una confianza alta para Iris-setosa.

## Solución de problemas

| Mensaje | Causa probable |
|---|---|
| No se encontró el archivo del modelo | Falta `modelo_iris.h5` en `models/` o el nombre no coincide |
| TensorFlow no está instalado | Falta ejecutar `pip install -r requirements.txt`, o el entorno virtual no está activado |
| `ModuleNotFoundError: No module named 'app'` | Se ejecutó desde otra carpeta; hay que estar en `iris_app/` al lanzar `python main.py` |
| `ModuleNotFoundError: No module named 'tkinter'` | En Linux: `sudo apt install python3-tk` |
| El modelo devolvió N salidas | La última capa de la red no tiene 3 neuronas, o `CLASS_NAMES` no coincide |
| Predice siempre la misma clase | Casi siempre es el punto 2 de arriba: falta aplicar el escalador |
