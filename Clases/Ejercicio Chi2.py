import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Datos de nicotina
data = np.array([1.09, 1.92, 2.31, 1.79, 2.28, 1.74, 1.47, 1.97,
                 0.85, 1.24, 1.58, 2.03, 1.70, 2.17, 2.55, 2.11,
                 1.86, 1.90, 1.68, 1.51, 1.64, 0.72, 1.69, 1.85,
                 1.82, 1.79, 2.46, 1.88, 2.08, 1.67, 1.37, 1.93,
                 1.40, 1.64, 2.09, 1.75, 1.63, 2.37, 1.75, 1.69])

data = np.sort(data)

alpha = 0.05
n = len(data)

# Intervalos
intervalos = np.array([stats.norm.ppf(i / (n + 1)) for i in range(1, n + 1)])

# Calcular la frecuencia esperada para cada intervalo
frec_esp = np.array([n / n] * n)

# Calcular el estadístico de chi2
chi2 = np.sum(((data - intervalos) ** 2) / frec_esp)

# Valor crítico de chi2 para alfa
v = n - 1
val_crit = stats.chi2.ppf(1 - alpha, v)

# Comparar el estadístico de chi-cuadrado con el valor crítico
if chi2 < val_crit:
    print("Los datos provienen de una distribución normal (no se rechaza H0).")
else:
    print("Los datos no provienen de una distribución normal (se rechaza H0).")

print("Estadístico de chi2:", chi2)
print("Valor crítico de chi2:", val_crit)

# Histograma
plt.hist(data, bins=6)
plt.title('Histograma de Contenido de Nicotina')
plt.xlabel('Contenido de Nicotina')
plt.ylabel('Frecuencia')
plt.show()