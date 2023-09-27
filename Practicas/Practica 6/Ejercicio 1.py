"""
Practica 6 y 7
- Encontrar el mejor modelo (aprox. R^2 = |1|)
- Reportar el código, valor de R^2 obtenido
- Graficar el histograma de los errores (media y ds del error)
- Concluir si los errores se aproximan a una distribución normal
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

y = [790, 1160, 929, 865, 1140, 929, 1109, 1365, 1112, 1150, 980, 990, 1112, 1252, 1326, 1330, 1365, 1280, 1119, 1328, 1584, 1428, 1365, 1415, 1415, 1465, 1490, 1725, 1523, 1705, 1605, 1746, 1235, 1390, 1405, 1395]
x = [99, 95, 95, 90, 105, 105, 90, 92, 98, 99, 99, 101, 99, 94, 97, 97, 99, 104, 104, 105, 94, 99, 99, 99, 99, 102, 104, 114, 109, 114, 115, 117, 104, 108, 109, 120]

plt.scatter(x, y)

model = np.polyfit(x, y, 12)

print("Modelo lineal", model)

"""
Predicción utilizando poly1d()
"""
predict = np.poly1d(model)

x_value = 36
predict(x_value)


"""
Exactitud del modelo R-squared
"""

predict = np.poly1d(model)
r2 = r2_score(y, predict(x))

print("El valor de r_square es", r2)

"""
Dibujando el modelo
"""

x_axis = range(90, 120)
y_axis = predict(x_axis)
plt.scatter(x, y)
plt.plot(x_axis, y_axis, c = "g")

mymodel = np.poly1d(np.polyfit(x, y, 3))

print("Mi modelo", mymodel)
predict3 = np.poly1d(mymodel)

####################################

"""
Calculo de los errores y-ypred
"""

errores = []
for n in y:
    errores = y - predict(x)
    
print("Errores: ", errores)

media_error = np.mean(errores)
print("Media de los errores = ", media_error)

ds_error = np.std(errores)
print("Desviación estándar de los errores = ", ds_error)

"""
Histograma de los errores
"""

plt.title("HISTOGRAMA DE LOS ERRORES")
plt.xlabel("CO2")
plt.ylabel("Weight")
plt.hist(errores)
plt.show()
