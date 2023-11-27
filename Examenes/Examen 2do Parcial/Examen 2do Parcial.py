# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 2023

@author: Fernando Haro Calvo
"""

import pandas as pd
from sklearn import linear_model

datos = pd.read_csv("heart.data_.cvs.csv")

x = datos[['biking', 'smoking']] # Variables independientes
y = datos['heart.disease'] # Variable dependiente

regr = linear_model.LinearRegression() # Obtiene el modelo de regresion lineal

regr.fit(x, y) # Realiza el calculo de regresion lineal con los parámetros x, y
print("Modelo:", regr.fit)

r2 = regr.score(x, y)
print("R2:", r2)
print("Interseccion:", regr.intercept_)
print("Slope:", regr.coef_)

print()
print("Modelo de regresion:")
print(f"y = {round(regr.intercept_, 4)} + ({round(regr.coef_[0], 4)})X0 + ({round(regr.coef_[1], 4)})X1 + e")

# Calcular la media de las variables independientes
print()
media_biking = x['biking'].mean()
media_smoking = x['smoking'].mean()
print(f"Media biking: {media_biking}")
print(f"Media smoking: {media_smoking}")
print()

predVal = regr.predict([[media_biking, media_smoking]])
print("Valor predicho:", predVal)