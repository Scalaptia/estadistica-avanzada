import numpy as np
from scipy.stats import f

data = np.array([
    [17.5, 16.9, 15.8, 18.6],
    [16.4, 19.2, 17.7, 15.4],
    [20.3, 15.7, 17.8, 18.9],
    [14.6, 16.7, 20.8, 18.9],
    [17.5, 19.2, 16.5, 20.5],
    [18.3, 16.2, 17.5, 20.1]
])

num_maquinas = 6
num_observaciones = 24

# Calcular la media total
media_total = np.mean(data)

# Calcular la suma de cuadrados totales (SCT)
sct = np.sum((data - media_total) ** 2)

# Calcular la media por máquina
media_por_maquina = np.mean(data, axis=0)

# Calcular la suma de cuadrados de tratamientos (STC)
stc = num_observaciones * np.sum((media_por_maquina - media_total) ** 2)

# Calcular la suma de cuadrados del error (SCE)
sce = sct - stc

# Grados de libertad
v = 5
v2 = 18

mct = stc / v
mce = sce / v

f_value = mct / mce
alpha = 0.05
f_critical = f.ppf(1 - alpha, v, v2)

# Imprimir resultados
print(f'(SCT): {sct}')
print(f'(STC): {stc}')
print(f'(SCE): {sce}')

print(f'F: {f_value}')
print(f'Valor crítico de F: {f_critical}')

# Comparar el valor p con el nivel de significancia
print()
if f_value < f_critical:
    print("Se rechaza la hipótesis nula")
else:
    print("No se rechaza la hipótesis nula")


