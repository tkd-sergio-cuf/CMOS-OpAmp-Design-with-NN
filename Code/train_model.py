import tensorflow as tf
from tensorflow import keras
!pip install -q -U keras-tuner
import numpy as np
import pandas as pd
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import PolynomialFeatures
from tensorflow.keras.layers import Dropout, Dense
from keras_tuner import RandomSearch
from sklearn.preprocessing import MinMaxScaler

np.random.seed(1234)
tf.random.set_seed(1234)

X = pd.read_csv('EntradasMAG.csv')
y = pd.read_csv('Salidas.csv')

# Escalar cada columna individualmente en X_train
X_scaled = X_train.copy()
scalers_X = {}

for column in X_scaled.columns:
    scaler = MinMaxScaler(feature_range=(0.01, 0.99))  # Ajustar el rango de normalización
    X_scaled[[column]] = scaler.fit_transform(X_scaled[[column]])  # Escalar la columna
    scalers_X[column] = scaler  # Guardar el escalador por si es necesario para nuevas predicciones

# Escalar cada columna individualmente en y_train
y_scaled = y_train.copy()
scalers_y = {}

for column in y_scaled.columns:
    scaler = MinMaxScaler(feature_range=(0.01, 0.99))  # Ajustar el rango de normalización
    y_scaled[[column]] = scaler.fit_transform(y_scaled[[column]])  # Escalar la columna
    scalers_y[column] = scaler  # Guardar el escalador para desescalar las predicciones

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_scaled)


input_dim = X_poly.shape[1]



oculta1 = tf.keras.layers.Dense(units=512, input_shape=[input_dim], activation="relu", kernel_regularizer=l2(2.170086313515314e-05), kernel_initializer='he_normal')
dropout1 = Dropout(0.30000000000000004)
oculta2 = tf.keras.layers.Dense(units=448, activation="relu", kernel_regularizer=l2(2.170086313515314e-05), kernel_initializer='he_normal') # en el libro lo ponen con 0.000013
dropout2 = Dropout(0.35)
oculta3 = tf.keras.layers.Dense(units=352, activation="relu", kernel_regularizer=l2(2.170086313515314e-05), kernel_initializer='he_normal')
dropout3 = Dropout(0.2)
oculta4 = tf.keras.layers.Dense(units=240, activation=tf.keras.activations.mish, kernel_regularizer=l2(2.170086313515314e-05), kernel_initializer='he_normal')
dropout4 = Dropout(0.45000000000000007)
salida = tf.keras.layers.Dense(units=9)
modelo = tf.keras.Sequential([oculta1, dropout1, oculta2, dropout2, oculta3, dropout3, oculta4, dropout4, salida])


modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.00019003838058425908),
    loss='mse',
    metrics=['mae']
)


historial = modelo.fit(X_poly, y_scaled, epochs=2000, validation_split=0.15, verbose=0)
print("Modelo entrenado!")

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])

# Gráfica para mae y val_mae
plt.figure(figsize=(10, 5))
plt.plot(historial.history['mae'], label='MAE')
plt.plot(historial.history['val_mae'], label='Validation_MAE')
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error")
plt.legend()
plt.show()
ultimo_val_mae = historial.history['val_mae'][-1]
print(ultimo_val_mae)