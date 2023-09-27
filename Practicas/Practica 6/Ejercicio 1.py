import numpy as np
import matplotlib.pylot as plt

x= [28, 8, 11, 37, 15, 25, 51, 11, 32, 34, 43, 2, 40, 16, 40, 25, 40, 17, 21, 57]
y= [8, 8, 9, 75, 22, 51 , 85, 4, 75, 48, 72, 1, 62, 37, 75, 42, 75, 47, 57, 95]

plt.scatter(x, y)

model = np.polyfit(x, y, 9)

print("Modelo lineal", model)

