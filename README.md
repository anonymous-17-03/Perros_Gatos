# 🐶🐱 Clasificación de imágenes (Perros y Gatos)

Este proyecto implementa un modelo de **inteligencia artificial con TensorFlow y Python**, entrenado para clasificar imágenes de **perros y gatos**.  
El modelo se exporta a los formatos **`.json`** y **`.bin`**, lo que permite ejecutarlo directamente en el navegador mediante **TensorFlow.js**, sin necesidad de un servidor backend.  

Puedes usarlo en tu computadora o en tu celular: solo apunta la cámara a un perro o un gato (puede ser una foto en pantalla, una imagen impresa o el animal real) y el sistema mostrará la predicción en tiempo real.

---

## 🚀 Cómo utilizarlo

### 1. Clonar el repositorio
Ejecuta en tu terminal:
```bash
git clone https://github.com/anonymous-17-03/Perros_Gatos.git
cd Perros_Gatos
````

### 2. Iniciar un servidor en la carpeta

Este proyecto utiliza TensorFlow\.js, por lo que los archivos deben servirse vía **HTTP/HTTPS** (no se pueden abrir directamente con doble clic).

Puedes usar varios servidores simples:

#### Con Python

```bash
python -m http.server 8000
```

#### Con PHP

```bash
php -S localhost:8000
```

Luego abre tu navegador en:
👉 [http://localhost:8000](http://localhost:8000)

---

## 📱 Uso

* Abre la página en tu navegador (PC o celular).
* Haz clic en **"Cambiar cámara"** para alternar entre cámara frontal y trasera (en caso de usar móvil).
* Apunta la cámara hacia un perro o un gato.
* En la parte inferior aparecerá la predicción en tiempo real.

---

## 📂 Archivos principales

* `model.json` y `group*-shard*.bin`: modelo entrenado en TensorFlow exportado a formato TensorFlow\.js.
* `index.html`: interfaz web para la clasificación.
* `Logo.png`, `favicon.ico`: recursos gráficos.

---

## ⚡ Tecnologías

* [TensorFlow](https://www.tensorflow.org/) / [TensorFlow.js](https://www.tensorflow.org/js)
* [Python](https://www.python.org/)
* [JavaScript](https://developer.mozilla.org/es/docs/Web/JavaScript)

