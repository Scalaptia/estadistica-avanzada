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

"""
Definir datos
"""
x = [99, 95, 95, 90, 105, 105, 90, 92, 98, 99, 99, 101, 99, 94, 97, 97, 99, 104, 104, 105, 94, 99, 99, 99, 99, 102, 104, 114, 109, 114, 115, 117, 104, 108, 109, 120]
y = [790, 1160, 929, 865, 1140, 929, 1109, 1365, 1112, 1150, 980, 990, 1112, 1252, 1326, 1330, 1365, 1280, 1119, 1328, 1584, 1428, 1365, 1415, 1415, 1465, 1490, 1725, 1523, 1705, 1605, 1746, 1235, 1390, 1405, 1395]

plt.figure(1)
plt.scatter(x, y)

"""
Definir modelo
"""
model = np.polyfit(x, y, 9) # Calcular polinomio de grado 9 (mejor valor de R^2)


"""
Calcular el valor de R^2
"""
predict = np.poly1d(model)
r2 = r2_score(y, predict(x)) # Calcular el valor de R^2
print("El valor de R-squared es:", r2)

"""
Graficar modelo
"""
x_axis = range(90, 120)
y_axis = predict(x_axis)
plt.plot(x_axis, y_axis, c="g")

#############
#  ERRORES  #
#############

"""
Calculo de los errores y-ypred
"""
errores = []
for n in y:
    errores = y - predict(x)

# print("Errores: ", errores)

media_error = np.mean(errores)
print("Media de los errores =", media_error)

ds_error = np.std(errores)
print("Desviación estándar de los errores =", ds_error)

"""
Graficar histograma de los errores
"""
plt.figure(2)
plt.title("HISTOGRAMA DE LOS ERRORES")
plt.xlabel("Errores")
plt.ylabel("Frecuencia")
plt.hist(errores, bins=9)
plt.show()
