import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

# ============================================================
# ⚙️ Ajustes iniciales
# ============================================================
TAMANO_IMG = 100
BATCH_SIZE = 32
EPOCHS = 30  # Ajusta según tu GPU

# Corrección temporal para cats_vs_dogs en TFDS
setattr(tfds.image_classification.cats_vs_dogs, '_URL',
        "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")

# ============================================================
# 📥 Descarga y preprocesamiento del dataset
# ============================================================
print("\n📥 Descargando y preparando dataset 'cats_vs_dogs'...\n")
datos, metadatos = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True)

def preprocesar(imagen, etiqueta):
    imagen = tf.image.resize(imagen, [TAMANO_IMG, TAMANO_IMG])
    imagen = tf.image.rgb_to_grayscale(imagen)
    imagen = tf.cast(imagen, tf.float32) / 255.0
    return imagen, etiqueta

train_size = int(0.85 * metadatos.splits['train'].num_examples)

train = datos['train'].take(train_size).map(preprocesar).shuffle(1000).batch(BATCH_SIZE)
val = datos['train'].skip(train_size).map(preprocesar).batch(BATCH_SIZE)

print("✅ Dataset cargado y dividido en entrenamiento y validación.\n")

# ============================================================
# 👀 Mostrar ejemplos
# ============================================================
for imagen, etiqueta in train.take(1):
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(tf.squeeze(imagen[i]), cmap="gray")
        plt.title("Perro" if etiqueta[i].numpy() == 1 else "Gato")
        plt.axis("off")
    plt.show()

# ============================================================
# 🧠 Definición de modelos
# ============================================================
print("\n🧠 Definiendo modelos...\n")

modeloDenso = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(TAMANO_IMG, TAMANO_IMG, 1)),
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

modeloCNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(TAMANO_IMG, TAMANO_IMG, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

modeloCNN2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(TAMANO_IMG, TAMANO_IMG, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(250, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# ============================================================
# ⚙️ Compilación
# ============================================================
for modelo in [modeloDenso, modeloCNN, modeloCNN2]:
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ============================================================
# 📊 Callbacks
# ============================================================
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

tensorboardDenso = TensorBoard(log_dir='logs/denso')
tensorboardCNN = TensorBoard(log_dir='logs/cnn')
tensorboardCNN2 = TensorBoard(log_dir='logs/cnn2')

# ✅ Guardar la mejor CNN2 automáticamente
checkpointCNN2 = ModelCheckpoint(
    filepath='mejor_modelo_cnn2.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ============================================================
# 🚀 Entrenamiento de modelos
# ============================================================
print("\n🚀 Entrenando Modelo Denso...\n")
hist_denso = modeloDenso.fit(train, validation_data=val,
                             epochs=EPOCHS, callbacks=[early_stop, tensorboardDenso])

print("\n🚀 Entrenando Modelo CNN...\n")
hist_cnn = modeloCNN.fit(train, validation_data=val,
                         epochs=EPOCHS, callbacks=[early_stop, tensorboardCNN])

print("\n🚀 Entrenando Modelo CNN2...\n")
hist_cnn2 = modeloCNN2.fit(train, validation_data=val,
                           epochs=EPOCHS, callbacks=[early_stop, tensorboardCNN2, checkpointCNN2])

# ============================================================
# 💾 Guardar modelos
# ============================================================
modeloDenso.save('perros_gatos_denso.h5')
print("✅ Modelo Denso guardado como perros_gatos_denso.h5")
modeloCNN.save('perros_gatos_cnn.h5')
print("✅ Modelo CNN guardado como perros_gatos_cnn.h5")
modeloCNN2.save('perros_gatos_cnn2.h5')
print("✅ Modelo CNN2 guardado como perros_gatos_cnn2.h5")

# ============================================================
# 📈 Visualización comparativa
# ============================================================
print("\n📈 Graficando desempeño de los modelos...\n")

plt.figure(figsize=(12,5))

# --- Precisión ---
plt.subplot(1,2,1)
plt.plot(hist_denso.history['val_accuracy'], label='Denso')
plt.plot(hist_cnn.history['val_accuracy'], label='CNN')
plt.plot(hist_cnn2.history['val_accuracy'], label='CNN2')
plt.title('Precisión en validación')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# --- Pérdida ---
plt.subplot(1,2,2)
plt.plot(hist_denso.history['val_loss'], label='Denso')
plt.plot(hist_cnn.history['val_loss'], label='CNN')
plt.plot(hist_cnn2.history['val_loss'], label='CNN2')
plt.title('Pérdida en validación')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
