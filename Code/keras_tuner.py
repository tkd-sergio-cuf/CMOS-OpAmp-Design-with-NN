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