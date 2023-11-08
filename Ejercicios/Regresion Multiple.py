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

predVal = regr.predict([[30.80124571, 10.89660802]])
print("Valor predicho:", predVal)
