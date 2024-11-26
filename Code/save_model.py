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