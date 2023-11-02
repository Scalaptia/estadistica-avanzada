from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

datos= [1.09, 1.92, 2.31, 1.79, 2.28, 1.74, 1.47, 1.97,
0.85, 1.24, 1.58, 2.03, 1.70, 2.17, 2.55, 2.11,
1.86, 1.90, 1.68, 1.51, 1.64, 0.72, 1.69, 1.85,
1.82, 1.79, 2.46, 1.88, 2.08, 1.67, 1.37, 1.93,
1.40, 1.64, 2.09, 1.75, 1.63, 2.37, 1.75, 1.69]

np.std(datos)

num_intervalos = 5
intervalos, bordes = np.histogram(datos, bins=num_intervalos)

for i in range(num_intervalos):
    print(f"Intervalo {i + 1}: {round(bordes[i], 2)} - {round(bordes[i + 1], 2)}, Frecuencia: {intervalos[i]}")

plt.hist(datos,bins=num_intervalos)

res = stats.normaltest(datos)
print("Valor X2: ", res.statistic)
