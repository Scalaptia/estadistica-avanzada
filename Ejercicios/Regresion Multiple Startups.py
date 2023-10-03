import pandas as pd
from sklearn import linear_model

datos = pd.read_csv("50_Startups.csv")
# print(datos)

# print(datos.columns) # Arreglo de las columnas
# print(datos.shape) # Tamaño de la matriz (tabla)

x = datos[['R&D Spend', 'Administration', 'Marketing Spend']] # Variables independientes
y = datos['Profit'] # Variable dependiente

regr = linear_model.LinearRegression() # Obtiene el modelo de regresion lineal

regr.fit(x, y) # Realiza el calculo de regresion lineal con los parámetros x, y
# print("MODELO", regr.fit)

r2 = regr.score(x, y)
print("Coefficient of determination (r2):", r2)

# Modelo
print("Slope (y):", regr.coef_)
print("Intercept:", regr.intercept_)

# Predice profit de un startup que gasta $131876 en R&D, $113867 en administración y $134050 en marketing
predictedProfit = regr.predict([[131876, 113867, 134050]])
print("Valor predicho:", predictedProfit)