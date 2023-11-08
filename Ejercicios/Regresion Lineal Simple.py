# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:32:21 2023

@author: ferna
"""

""" Walpole problema 11.1 """

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

x = [ 17.3, 19.3, 19.5, 19.7, 22.9, 23.1, 26.4, 26.8, 27.6, 28.1, 28.2, 28.7, 29, 29.6, 29.9, 29.9, 30.3, 31.3, 36, 39.5, 40.4, 44.3, 44.6, 50.4, 55.9]
y = [ 71.7, 48.3, 88.3, 75, 91.7, 100, 73.3, 65, 75, 88.3, 68.3, 96.7, 76.7, 78.3, 60, 71.7, 85, 85, 88.3, 100, 100, 100, 91.7, 100, 71.7 ]

plt.scatter(x, y)

model = np.polyfit(x, y, 2)
print("Modelo Lineal:", model)

predict=np.poly1d(model)

x_val = 30
print(f"Predicion para x = {x_val}: {predict(x_val)}")

r2 = r2_score(y, predict(x))
print(f"El valor de R2 es: {r2}")

