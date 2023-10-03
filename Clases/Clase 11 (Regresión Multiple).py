import pandas as pd
from sklearn import linear_model

datos = pd.read_csv("cars.csv")
# print(datos)

# print(datos.columns) # Arreglo de las columnas
# print(datos.shape) # Tamaño de la matriz (tabla)

x = datos[['Weight', 'Volume']] # Variables independientes
y = datos['CO2'] # Variable dependiente

regr = linear_model.LinearRegression() # Obtiene el modelo de regresion lineal

regr.fit(x, y) # Realiza el calculo de regresion lineal con los parámetros x, y
# print("MODELO", regr.fit)

r2 = regr.score(x, y)
print("Coefficient of determination:", r2)
print("Intercept:", regr.intercept_)
print("Slope:", regr.coef_)

# Predice emision de CO2 de un carro de 2300kg con motor 1300cm3
predictedCO2 = regr.predict([[2300, 1300]])
print("Valor predicho:", predictedCO2)