from scipy.stats import chi2_contingency

datos = [[15, 29], [27, 19]]

# Chi^2
chi2, p, x, x = chi2_contingency(datos)
alpha = 0.01

print(f"Chi^2: {chi2}")
print(f"Valor p: {p}")

print()
if p < alpha:
    print("Se rechaza la hipótesis nula.")
else:
    print("No se rechaza la hipótesis nula.")
