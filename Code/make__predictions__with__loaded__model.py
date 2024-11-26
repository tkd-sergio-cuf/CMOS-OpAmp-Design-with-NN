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