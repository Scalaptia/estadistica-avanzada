import numpy as np
from scipy.stats import f_oneway

datos = np.array([
    [5.2, 9.1, 3.2, 2.4, 7.1],
    [4.7, 7.1, 5.8, 3.4, 6.6],
    [8.1, 8.2, 2.2, 4.1, 9.3],
    [6.2, 6.0, 3.1, 1.0, 4.2],
    [3.0, 9.1, 7.2, 4.0, 7.6]
])

# ANOVA
anova, p = f_oneway(*datos.T)
nivel_significancia = 0.05

# Imprimir resultados
print(f"ANOVA: {anova}")
print(f"Valor p: {p}")

print()
if p < nivel_significancia:
    print("Se rechaza la hipótesis nula.")
else:
    print("No se rechaza la hipótesis nula.")