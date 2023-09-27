# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:17:10 2023

@author: Fernando Haro Calvo
"""

import numpy as np
import matplotlib.pyplot as plt

x= [28, 8, 11, 37, 15, 25, 51, 11, 32, 34, 43, 2, 40, 16, 40, 25, 40, 17, 21, 57]
y= [8, 8, 9, 75, 22, 51 , 85, 4, 75, 48, 72, 1, 62, 37, 75, 42, 75, 47, 57, 95]

plt.scatter(x, y)

model = np.polyfit(x, y, 9)

print("Modelo lineal", model)

"""
Predicción utilizando poly1d()
"""
predict = np.poly1d(model)

x_value = 20
predict(x_value)

"""
Exactitud del modelo R-squared
Sse debe importar sklearn.metric import r2_score
"""
from sklearn.metrics import r2_score
r2 = r2_score(y, predict(x))

print("El valor de r_square es", r2) #coeficiente de determinacion

"""
Dibujando el modelo
"""

x_axis = range(0, 60)
y_axis = predict(x_axis)
plt.scatter(x, y)
plt.plot(x_axis, y_axis, c = "g")

#### Aqui termina el ejercicio para polinomio de grado 9

mymodel = np.poly1d(np.polyfit(x, y, 3))

print("Mi modelo", mymodel)
predict = np.poly1d(mymodel)

x_value = 20
x_20 = predict(x_value)

print("Valor de y para x=20 es: ", x_20)

r2 = r2_score(list(x_20), x_value)
print("El valor de r_square es", r2) #coeficiente de determinacion
