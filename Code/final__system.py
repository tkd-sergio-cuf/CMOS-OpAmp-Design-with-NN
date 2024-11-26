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