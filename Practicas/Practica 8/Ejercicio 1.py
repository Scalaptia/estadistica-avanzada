import pandas as pd
from sklearn import linear_model

datos = pd.read_csv("global-data-on-sustainable-energy.csv")
# print(datos)

# print(datos.columns) # Arreglo de las columnas
# print(datos.shape) # Tamaño de la matriz (tabla)

# Variables independientes
x = datos[['Access to electricity (% of population)',
           'Primary energy consumption per capita (kWh/person)',
           'Energy intensity level of primary energy (MJ/$2017 PPP GDP)',
           'Value_co2_emissions_kt_by_country']]

y = datos['gdp_per_capita'] # Variable dependiente

regr = linear_model.LinearRegression() # Obtiene el modelo de regresion lineal

regr.fit(x, y) # Realiza el calculo de regresion lineal con los parámetros x, y
# print("MODELO", regr.fit)

r2 = regr.score(x, y)
print("Coefficient of determination:", r2)
print("Intercept:", regr.intercept_)
print("Slope:", regr.coef_)