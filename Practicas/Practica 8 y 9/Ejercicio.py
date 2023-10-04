# Datos de 395 estudiantes para predecir su rendimiento en base a factores
# de su vida personal.

import numpy as np
import pandas as pd
from sklearn import linear_model

datos = pd.read_csv("student-mat.csv")
# print(datos)

# print(datos.columns) # Arreglo de las columnas
# print(datos.shape) # Tamaño de la matriz (tabla)

# Variables independientes
x = datos[['traveltime', #  El tiempo de viaje desde casa a la escuela (1-4)
           'studytime', # El tiempo de estudio semanal del estudiante (1-4)
           'famrel', # La calidad de las relaciones familiares (1-5)
           'freetime', # Tiempo libre fuera de la escuela (1-5)
           'health' # Estado de salud del estudiante (1-5)
         ]]

# G1, G2 Y G3 siendo calificaciones parciales (0-20)
y = (datos[('G1')] + datos[('G2')] + datos[('G3')]) / 60 * 100 # Variable dependiente (PROMEDIO DE LOS 3 PERIODOS)

regr = linear_model.LinearRegression() # Obtiene el modelo de regresion lineal

regr.fit(x, y) # Realiza el calculo de regresion lineal con los parámetros x, y

r2 = regr.score(x, y)
print("Coefficient of determination (r2):", r2)

# Modelo
print("Slope (y):", regr.coef_)
print("Intercept:", regr.intercept_)

# Predice promedio de un estudiante los siguientes datos:
# - 1 Tiempo de viaje de casa a la escuela
# - 4 Tiempo de estudio semanal
# - 5 Calidad de relacion familiar
# - 5 Tiempo libre fuera de la escuela
# - 5 Estado de salud
predictedProfit = regr.predict([[1, 4, 5, 5, 5]])
print("\nPromedio General:", round(np.mean(y), 2), "de 100")
print("Promedio predicho del estudiante:", round(predictedProfit[0], 2), "de 100\n")