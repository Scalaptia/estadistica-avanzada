import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

x = [10, 10 , 10, 10, 10, 50, 50, 50, 50, 50]
y = [13, 18, 16, 15, 20, 86, 90, 88, 88, 92]

plt.scatter (x, y)
plt.plot(x, y, c = "g")

mymodel = np.poly1d(np.polyfit(x, y, 1))

predict = np.poly1d(mymodel)
x_value = 54
x_54 = predict(x_value)

print("Valor de Y para x=54 ", x_54)

#### Exactitud del modelo R-squared

r2 = r2_score(y,predict(x)) # COEFICIENTE DE DETERMINACION
print("El valor de r_square es", r2)