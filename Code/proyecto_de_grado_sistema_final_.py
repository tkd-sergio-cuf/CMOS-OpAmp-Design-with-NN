
import tensorflow as tf
from tensorflow import keras
!pip install -q -U keras-tuner
import numpy as np
import pandas as pd
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from tensorflow.keras.layers import Dropout, Dense
from keras_tuner import RandomSearch

scaler = StandardScaler()

np.random.seed(1234)
tf.random.set_seed(1234)

X_train = pd.read_csv('Entradas.csv')
y_train = pd.read_csv('Salidas.csv')
X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_np)

X_train_scaled = scaler.fit_transform(X_train_poly)


def build_model(hp):
    input_dim = X_train_scaled.shape[1]

    modelo = tf.keras.Sequential()

    l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-2, sampling='log')

    modelo.add(Dense(units=hp.Int('units_1', min_value=128, max_value=512, step=32),
                     input_shape=[input_dim],
                     activation=hp.Choice('activation_1', values=['mish', 'relu']),
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                     kernel_initializer='he_normal'))


    modelo.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.05)))


    modelo.add(Dense(units=hp.Int('units_2', min_value=64, max_value=512, step=32),
                     activation=hp.Choice('activation_2', values=['mish', 'relu']),
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                     kernel_initializer='he_normal'))


    modelo.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.05)))


    modelo.add(Dense(units=hp.Int('units_3', min_value=32, max_value=512, step=32),
                     activation=hp.Choice('activation_3', values=['mish', 'relu']),
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                     kernel_initializer='he_normal'))


    modelo.add(Dropout(hp.Float('dropout_3', min_value=0.1, max_value=0.5, step=0.05)))


    modelo.add(Dense(units=hp.Int('units_4', min_value=16, max_value=512, step=32),
                     activation=hp.Choice('activation_4', values=['mish', 'relu']),
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                     kernel_initializer='he_normal'))


    modelo.add(Dropout(hp.Float('dropout_4', min_value=0.1, max_value=0.5, step=0.05)))


    modelo.add(Dense(units=9))


    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')),
        loss='mse',
        metrics=['mae']
    )

    return modelo

# Instanciar el tuner
tuner = RandomSearch(
    build_model,
    objective='val_mae',
    max_trials=20,  # Número de combinaciones de hiperparámetros a probar
    executions_per_trial=1,  # Número de veces que se entrena cada modelo
    directory='my_dir',
    project_name='hyperparametros'
)

early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitorea la pérdida en el conjunto de validación
    patience=50,  # Número de épocas sin mejora antes de detener el entrenamiento
    restore_best_weights=True  # Restaura los mejores pesos al finalizar
)

tuner.search(X_train_scaled, y_train_np,
             epochs=500,
             validation_split=0.15,
             callbacks=[early_stopping],
             verbose=0)

# Obtener el mejor modelo
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Imprimir el mejor val_mae al final
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Mejor val_mae encontrado: {best_hps.values}")

best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
best_val_mae = best_trial.metrics.get_best_value('val_mae')

# Imprimir el mejor val_mae
print(f"Mejor val_mae obtenido: {best_val_mae}")

"""# Train model"""

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

"""# Save model"""

import pickle
 #-------- GUARDAR EL MODELO UNA VEZ HECHA LA PREDICCIÓN --------

# Guardar el modelo en formato HDF5
modelo.save('/content/drive/MyDrive/Tesis/Modelo_Redes/CD/Modelo/FINAL_500_DATOS.h5')
# modelo.save('/content/drive/MyDrive/Datos_CT_Finales/modelo_entrenado_Prueba_1500.h5')

# Guardar el escalador y las características polinómicas
with open('/content/drive/MyDrive/Tesis/Modelo_Redes/CD/Modelo/FINAL_500_DATOS_Scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('/content/drive/MyDrive/Tesis/Modelo_Redes/CD/Modelo/FINAL_500_DATOS_Poly.pkl', 'wb') as f:
    pickle.dump(poly, f)

print("Modelo, escalador y características polinómicas guardados correctamente.")

"""# Load model"""

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

"""# Make predictions with loaded model"""

X_nuevos_no_escalados = np.array([
  [13.2E-06, 11926.14, 69.22, 16595.86, 2675740, 1.408, 240713.25, 2.37],
])

# Crear una copia escalada de los nuevos datos usando los escaladores guardados
X_nuevos_escalados = X_nuevos_no_escalados.copy()

# Escalar los nuevos datos columna por columna
for i, column in enumerate(X_train.columns):
    X_nuevos_escalados[:, i] = scalers_X[column].transform(X_nuevos_no_escalados[:, i].reshape(-1, 1)).flatten()

# Crear características polinomiales para los nuevos datos escalados
X_nuevos_polynomial = poly.transform(X_nuevos_escalados)

# Realizar la predicción
prediccion_escalada = modelo_cargado.predict(X_nuevos_polynomial)

# Desescalar la predicción para obtener los valores originales
prediccion_desescalada = prediccion_escalada.copy()

for i, column in enumerate(y_train.columns):
    prediccion_desescalada[:, i] = scalers_y[column].inverse_transform(prediccion_escalada[:, i].reshape(-1, 1)).flatten()

# Imprimir las predicciones finales
#print(f"Predicciones: {prediccion_desescalada}")

# Nombres de las variables correspondientes a los valores predichos
nombres_variables = ['W3', 'W4', 'Vpol1_DC', 'Vpol2_DC', 'Vpol3_DC', 'W6', 'W2', 'W1', 'W5']

# Iterar sobre las 10 predicciones
for i, prediccion in enumerate(prediccion_desescalada):
    print(f"Predicción {i + 1}:")
    print(X_nuevos_no_escalados[i][0])
    # Imprimir cada valor junto con el nombre de la variable en notación científica
    for j, valor in enumerate(prediccion):
        print(f"{nombres_variables[j]}: {valor:.4e}")  # Notación científica con 10 decimales

    print()  # Salto de línea entre predicciones

"""# Final System



"""

import tensorflow as tf
import numpy as np
import pandas as pd
import time
import warnings
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from tensorflow.keras.losses import mse
from IPython.display import clear_output, display
import ipywidgets as widgets
from google.colab import output
import gc

# Optimizar configuración de TensorFlow para Colab
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Configuración inicial y supresión de warnings
warnings.filterwarnings("ignore", category=UserWarning)
tf.get_logger().setLevel('ERROR')
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

class ModeloPredictivo:
    def __init__(self):
        self.modelos = {}
        self.configuraciones = {}
        self.cache_scalers = {}
        self.cache_poly = {}

    def limpiar_memoria(self):
        """Limpia la memoria de manera segura"""
        gc.collect()
        tf.keras.backend.clear_session()

    def cargar_escalar_datos(self, ruta_entrada, ruta_salida, rango=(0.01, 0.99)):
        """Carga y escala datos con manejo de caché"""
        cache_key = f"{ruta_entrada}_{ruta_salida}"

        if cache_key in self.cache_scalers:
            return self.cache_scalers[cache_key]

        X = pd.read_csv(ruta_entrada)
        y = pd.read_csv(ruta_salida)

        X_scaled = X.copy()
        y_scaled = y.copy()

        scalers_X, scalers_y = {}, {}

        for df, scalers, prefix in [(X, scalers_X, 'X'), (y, scalers_y, 'y')]:
            for column in df.columns:
                scaler = MinMaxScaler(feature_range=rango)
                if prefix == 'X':
                    X_scaled[[column]] = scaler.fit_transform(df[[column]])
                else:
                    y_scaled[[column]] = scaler.fit_transform(df[[column]])
                scalers[column] = scaler

        result = (X, X_scaled, y, y_scaled, scalers_X, scalers_y)
        self.cache_scalers[cache_key] = result
        return result

    def cargar_modelos(self):
        """Carga modelos con lazy loading"""
        modelos_config = {
            'cascodo_doblado': {
                'ruta_modelo': './drive/MyDrive/Tesis/Modelo_Redes/CD/Modelo/FINAL_500_DATOS.h5',
                'ruta_entrada': './drive/MyDrive/Tesis/Modelo_Redes/CD/Datos/CD-Entrada.csv',
                'ruta_salida': './drive/MyDrive/Tesis/Modelo_Redes/CD/Datos/CD-Salida.csv',
                'grado_poly': 2,
                'variables': ['W3', 'W4', 'Vpol1_DC', 'Vpol2_DC', 'Vpol3_DC', 'W6', 'W2', 'W1', 'W5'],
                'orden_entrada': ['Iref', 'PSRR', 'Margen de fase', 'Ganancia', 'FGU', 'DR', 'CMRR', 'SR']
            },
            'cascodo_telescopico': {
                'ruta_modelo': './drive/MyDrive/Tesis/Modelo_Redes/CT/Modelo/Red_neuronal_CT_Final.h5',
                'ruta_entrada': './drive/MyDrive/Tesis/Modelo_Redes/CT/Datos/Aleatorios_500 - Entrada.csv',
                'ruta_salida': './drive/MyDrive/Tesis/Modelo_Redes/CT/Datos/Aleatorios_500 - Salida.csv',
                'grado_poly': 3,
                'variables': ['Vpol1_DC', 'Vpol2_DC', 'W1', 'W2', 'W3', 'W4', 'W5'],
                'orden_entrada': ['Iref', 'Ganancia', 'CMRR', 'PSRR', 'FGU', 'Margen de fase', 'SR', 'DR']
            }
        }

        self.configuraciones = modelos_config

    def get_modelo(self, tipo_modelo):
        """Obtiene modelo con lazy loading"""
        if tipo_modelo not in self.modelos:
            config = self.configuraciones[tipo_modelo]
            self.modelos[tipo_modelo] = tf.keras.models.load_model(
                config['ruta_modelo'],
                custom_objects={'mse': mse}
            )
        return self.modelos[tipo_modelo]

    def hacer_prediccion(self, modelo, X_nuevos_no_escalados, scalers_X, poly, X_train, y_train, scalers_y):
        """Función de predicción"""
        X_nuevos_no_escalados = np.array(X_nuevos_no_escalados)

        X_nuevos_escalados = np.zeros_like(X_nuevos_no_escalados, dtype=np.float32)
        for i, column in enumerate(X_train.columns):
            X_nuevos_escalados[:, i] = scalers_X[column].transform(
                X_nuevos_no_escalados[:, i].reshape(-1, 1)
            ).flatten()

        X_nuevos_polynomial = poly.transform(X_nuevos_escalados)
        prediccion_escalada = modelo.predict(X_nuevos_polynomial, verbose=0)

        prediccion_desescalada = np.zeros_like(prediccion_escalada, dtype=np.float32)
        for i, column in enumerate(y_train.columns):
            prediccion_desescalada[:, i] = scalers_y[column].inverse_transform(
                prediccion_escalada[:, i].reshape(-1, 1)
            ).flatten()

        return prediccion_desescalada

class InterfazUsuario:
    def __init__(self, modelo_predictivo):
        self.modelo_predictivo = modelo_predictivo
        self.setup_widgets()

    def reset_widgets(self):
        """Reinicia los valores de los widgets a su estado inicial"""
        self.modelo_selector.value = 'cascodo_doblado'
        for widget in self.variables_input.values():
            widget.value = 0.0

    def setup_widgets(self):
        """Configura los widgets con estilo mejorado"""
        style = {'description_width': '120px'}
        layout = widgets.Layout(width='300px')

        self.modelo_selector = widgets.Dropdown(
            options=[
                ('Modelo Cascodo Doblado', 'cascodo_doblado'),
                ('Modelo Cascodo Telescópico', 'cascodo_telescopico')
            ],
            value='cascodo_doblado',
            description='Modelo:',
            style=style,
            layout=layout
        )

        self.variables_input = {
            'Ganancia': widgets.FloatText(description='Ganancia:', value=0.0, style=style, layout=layout),
            'Iref': widgets.FloatText(description='Iref:', value=0.0, style=style, layout=layout),
            'PSRR': widgets.FloatText(description='PSRR:', value=0.0, style=style, layout=layout),
            'Margen de fase': widgets.FloatText(description='MF:', value=0.0, style=style, layout=layout),
            'FGU': widgets.FloatText(description='FGU:', value=0.0, style=style, layout=layout),
            'DR': widgets.FloatText(description='DR:', value=0.0, style=style, layout=layout),
            'CMRR': widgets.FloatText(description='CMRR:', value=0.0, style=style, layout=layout),
            'SR': widgets.FloatText(description='SR:', value=0.0, style=style, layout=layout)
        }

        self.boton_prediccion = widgets.Button(
            description="Hacer Predicción",
            button_style='primary',
            layout=widgets.Layout(width='150px')
        )
        self.boton_reiniciar = widgets.Button(
            description="Nueva Predicción",
            button_style='info',
            layout=widgets.Layout(width='150px')
        )
        self.boton_salir = widgets.Button(
            description="Salir",
            button_style='danger',
            layout=widgets.Layout(width='150px')
        )

        self.configurar_eventos()

    def configurar_eventos(self):
        """Configura los manejadores de eventos"""
        self.boton_prediccion.on_click(self.on_boton_prediccion_click)
        self.boton_reiniciar.on_click(self.on_boton_reiniciar_click)
        self.boton_salir.on_click(self.on_boton_salir_click)

    def on_boton_prediccion_click(self, b):
        """Manejador del botón de predicción"""
        try:
            clear_output(wait=True)

            tipo_modelo = self.modelo_selector.value
            config = self.modelo_predictivo.configuraciones[tipo_modelo]

            orden_entrada = [self.variables_input[var].value for var in config['orden_entrada']]
            X_nuevos = np.array([orden_entrada], dtype=np.float32)

            modelo = self.modelo_predictivo.get_modelo(tipo_modelo)
            datos = self.modelo_predictivo.cargar_escalar_datos(
                config['ruta_entrada'],
                config['ruta_salida']
            )

            poly = PolynomialFeatures(degree=config['grado_poly'])
            poly.fit(datos[1])

            start_time = time.time()
            prediccion = self.modelo_predictivo.hacer_prediccion(
                modelo, X_nuevos, datos[4],
                poly, datos[0], datos[2], datos[5]
            )
            tiempo = time.time() - start_time

            print("\nPredicción:")
            for nombre, valor in zip(config['variables'], prediccion[0]):
                if 'W' in nombre:
                    print(f"{nombre}: {(valor*1000000):#.4f} µm")
                else:
                    print(f"{nombre}: {valor:.4f} V")
            print(f"\nTiempo de predicción: {tiempo:.4f} segundos")

            display(widgets.HBox([self.boton_reiniciar, self.boton_salir]))

        except Exception as e:
            print(f"Error durante la predicción: {str(e)}")
            display(self.boton_reiniciar)

    def on_boton_reiniciar_click(self, b):
        """Manejador del botón de reinicio"""
        clear_output(wait=True)
        self.reset_widgets()
        self.mostrar_formulario()
        self.modelo_predictivo.limpiar_memoria()

    def on_boton_salir_click(self, b):
        """Manejador del botón de salida"""
        clear_output(wait=True)
        print("Programa terminado.")
        self.modelo_predictivo.limpiar_memoria()

    def mostrar_formulario(self):
        """Muestra el formulario con layout mejorado"""
        display(widgets.VBox([
            self.modelo_selector,
            *self.variables_input.values(),
            widgets.HBox([self.boton_prediccion])
        ]))

def main():
    """Función principal de inicialización"""
    modelo_predictivo = ModeloPredictivo()
    modelo_predictivo.cargar_modelos()
    interfaz = InterfazUsuario(modelo_predictivo)
    interfaz.mostrar_formulario()

if __name__ == "__main__":
    main()
