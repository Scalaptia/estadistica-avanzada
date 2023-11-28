import numpy as np
from scipy.stats import norm, chi2


datos = np.array([23, 60, 79, 32, 57, 74, 52, 70, 82, 36, 80, 77, 81, 95, 41, 65, 92, 85, 55, 76, 52,
                  10, 64, 75, 78, 25, 80, 98, 81, 67, 41, 71, 83, 54, 64, 72, 88, 62, 74, 43, 60, 78,
                  89, 76, 84, 48, 84, 90, 15, 79, 34, 67, 17, 82, 69, 74, 63, 80, 85, 61])

intervalos = [0, 40, 60, 80, 100]
mu = 65
sigma = 21

# Frec obs
frec_obs, bordes = np.histogram(datos, bins=intervalos)

# Frec esp
prob_acum = np.diff(norm.cdf(bordes, mu, sigma))
frec_esp = len(datos) * prob_acum

# Chi^2
chi_cuadrado = np.sum((frec_obs - frec_esp)**2 / frec_esp)

g_lib = len(intervalos) - 2
valor_critico = chi2.ppf(0.95, g_lib)


print(f"Frecuencias observadas: {frec_obs}")
print(f"Frecuencias esperadas: {frec_esp}")
print(f"Chi^2: {chi_cuadrado}")
print(f"Grados de libertad: {g_lib}")
print(f"Valor critico: {valor_critico}")

print()
if chi_cuadrado > valor_critico:
    print("Se rechaza la hipótesis nula.")
else:
    print("No se rechaza la hipótesis nula.")