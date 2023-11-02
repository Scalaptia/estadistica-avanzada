# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:30:17 2023

@author: jinh2023

scipy.stats.normaltest(a, axis=0, nan_policy='propagate')[source]

Returns:
statisticfloat or array
s^2 + k^2, where s is the z-score returned by skewtest and k is the z-score returned by kurtosistest.

pvaluefloat or array

"""

from scipy import stats

import matplotlib.pyplot as plt
datos=[2.2, 4.1, 3.5, 4.5, 3.2, 3.7, 3, 2.6,
       3.4, 1.6, 3.1, 3.3, 3.8, 3.1, 4.7, 3.7,
       2.5, 4.3, 3.4, 3.6, 2.9, 3.3, 3.9, 3.1,
       3.3, 3.1, 3.7, 4.4, 3.2, 4.1, 1.9, 3.4,
       4.7, 3.8, 3.2, 2.6, 3.9, 3, 4.2, 3.5]

plt.hist(datos,bins=7)
res = stats.normaltest(datos)

print(res.statistic)
# Out[5]: 1.5096401543334301

print(res.pvalue)
# Out[6]: 0.4700951879801155
