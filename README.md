# 🐶🐱 Clasificación de Imágenes de Perros y Gatos

Este proyecto implementa un **modelo de inteligencia artificial** usando **TensorFlow y Python**, entrenado para **clasificar imágenes de perros y gatos**. El modelo también se puede **exportar a TensorFlow\.js**, permitiendo su uso en navegadores sin necesidad de un servidor backend.

El proyecto incluye tres modelos:

* **Denso**: red completamente conectada (MLP).
* **CNN**: red convolucional básica.
* **CNN2**: CNN con Dropout y capa densa más grande, normalmente el modelo con mejor desempeño.

Al final del entrenamiento, todos los modelos se guardan, pero el **CNN2** se guarda también automáticamente en su **mejor versión según validación**.

---

## 🛠️ Requisitos y entorno

### 1. Crear un entorno virtual (recomendado)

```bash
# Crear entorno virtual llamado venv
python3 -m venv venv

# Activar entorno en Linux/macOS
source venv/bin/activate

# Activar entorno en Windows
venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install --upgrade pip
pip install tensorflow tensorflow-datasets matplotlib tensorflowjs
```

---

## 🚀 Uso del proyecto

### 1. Clonar repositorio

```bash
git clone https://github.com/anonymous-17-03/Perros_Gatos.git
cd Perros_Gatos
```

### 2. Entrenamiento del modelo

```bash
# Ejecutar script de entrenamiento
python entrenar_modelo.py
```

Este script realiza lo siguiente:

1. Descarga y preprocesa el dataset `cats_vs_dogs` de TensorFlow Datasets.
2. Define y compila tres modelos diferentes: Denso, CNN y CNN2.
3. Entrena cada modelo usando **EarlyStopping** para evitar sobreentrenamiento.
4. Guarda los tres modelos en formato `.h5`.
5. Grafica la precisión y pérdida de validación de los tres modelos para comparación visual.

### 3. Exportar modelo CNN2 a TensorFlow\.js

```bash
tensorflowjs_converter --input_format=keras \
    perros_gatos_cnn2.h5 \
    tfjs_model/
```

Esto generará una carpeta `tfjs_model` con los archivos necesarios (`model.json` y `group*-shard*.bin`) para usar el modelo directamente en la web.

---

### 4. Servir la interfaz web

Se proporciona un ejemplo en `index.html`. Puedes usar un servidor HTTP simple:

#### Con Python:

```bash
python -m http.server 8000
```

#### Con PHP:

```bash
php -S localhost:8000
```

Luego abre el navegador en:

```
http://localhost:8000
```

---

## 📚 Explicación de bloques del código

1. **Descarga y preprocesamiento del dataset**

```python
datos, metadatos = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True)
```

* Descarga automáticamente el dataset de Kaggle.
* Convierte las imágenes a **grayscale** y normaliza los píxeles a `[0,1]`.
* Divide el dataset en **train** (85%) y **validation** (15%).

> ⚠️ `grayscale` reduce complejidad y memoria. La CNN sigue aprendiendo patrones de forma eficiente.

---

2. **Definición de modelos**

* **Denso**: todo conectado, bueno para problemas pequeños pero limitado en imágenes.
* **CNN**: utiliza capas convolucionales para extraer características espaciales de las imágenes.
* **CNN2**: CNN con **Dropout** (previene overfitting) y capa densa más grande (250 neuronas), normalmente con mejor desempeño.

---

3. **Compilación**

```python
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

* **Optimizer Adam**: ajusta los pesos de la red automáticamente.
* **Binary crossentropy**: función de pérdida para problemas de clasificación binaria.
* **Accuracy**: métrica para evaluar desempeño en clasificación.

---

4. **Callbacks**

* **EarlyStopping**: detiene entrenamiento si la validación no mejora, evitando sobreajuste y ahorro de tiempo.
* **ModelCheckpoint**: guarda automáticamente la mejor versión del modelo CNN2 según validación.
* **TensorBoard**: permite visualizar gráficas de entrenamiento en tiempo real.

---

5. **Entrenamiento y gráficos**

* Entrena los modelos y genera **gráficas comparativas** de precisión y pérdida en validación.
* Esto permite ver visualmente que **CNN2 suele ser superior** a Denso y CNN simples.

---

6. **En caso de trabajar en Colab**

En una celda nueva, escribe:

```python
from google.colab import files
```

Esta librería permite descargar archivos directamente desde el entorno de Colab.

```python
files.download("perros_gatos_denso.h5")
files.download("perros_gatos_cnn.h5")
files.download("perros_gatos_cnn2.h5")
files.download("mejor_modelo_cnn2.h5")
```

💡 Esto abrirá automáticamente la ventana de descarga en tu navegador para cada archivo.

---

## ⚙️ Explicación de términos técnicos

* **CNN (Convolutional Neural Network)**: red neuronal especializada en procesar datos con estructura de cuadrícula (imágenes).
* **Dropout**: técnica que desactiva neuronas aleatorias durante entrenamiento para mejorar generalización.
* **EarlyStopping**: detiene el entrenamiento cuando no hay mejoras en la validación, evitando overfitting.
* **Overfitting**: cuando el modelo aprende demasiado los datos de entrenamiento y falla en generalizar a nuevos datos.
* **Normalización**: ajustar valores de píxeles de `[0,255]` a `[0,1]` para que el entrenamiento sea más estable.

---

## ✅ Conclusión

Este proyecto demuestra cómo construir y entrenar **modelos de IA para clasificación de imágenes**, mostrando:

* Cómo preprocesar datos con TensorFlow Datasets.
* Diferencias entre **redes densas y convolucionales**.
* Cómo **guardar modelos, aplicar callbacks y visualizar métricas**.
* Cómo convertir el modelo entrenado a **TensorFlow\.js** para uso en web.

La combinación de **CNN2 + Dropout** ofrece un modelo balanceado, eficiente y listo para producción.
