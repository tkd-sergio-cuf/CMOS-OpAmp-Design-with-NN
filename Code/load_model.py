import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pickle  # Para cargar el escalador y las características polinómicas
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import mse # Import the mse loss function

# Configuración de impresión completa para NumPy
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


X_train = pd.read_csv('EntradasMAG.csv')
y_train = pd.read_csv('Salidas.csv')
X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()


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

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_scaled)


# -------- CARGAR EL MODELO GUARDADO --------

modelo_cargado = tf.keras.models.load_model('/content/drive/MyDrive/Tesis/Modelos Finales/FINAL_TODOS_DATOS.h5', custom_objects={'mse': mse})

# Cargar el escalador y las características polinómicas desde los archivos guardados
with open('/content/drive/MyDrive/Tesis/Modelos Finales/FINAL_TODOS_DATOS_Scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('/content/drive/MyDrive/Tesis/Modelos Finales/FINAL_TODOS_DATOS_Poly.pkl', 'rb') as f:
    poly = pickle.load(f)