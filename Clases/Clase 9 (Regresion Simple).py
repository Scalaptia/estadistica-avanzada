import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

### Regresion Simple ###

x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6] # Numero de Vendedores
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86] # Ventas en Miles

plt.scatter(x, y)
plt.title('Regresión Vendedores y Ventas')
plt.xlabel('Numero de Vendedores')
plt.ylabel('Ventas en Miles')
plt.show()

# Para graficar los valores yi predichos por el modelo
slope, intercept, r, p, std_err, intercept_std_err = stats.linregress(x, y)

def myFunc(x):
    return slope * x + intercept

mymodel = list(map(myFunc, x))

y_de_10 = myFunc(10)

print("El valor de y cuando es x igual a diez", y_de_10)

### Regresion Lineal (Numpy) ###

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
mymodel1 = np.poly1d(np.polyfit(x, y, 3))

print("Coeficientes del polinomio de orden 3: ", mymodel1)

## r = ??

## y de 10 apra mymodel1 = ??