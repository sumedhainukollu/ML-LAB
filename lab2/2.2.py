import numpy as np

x = np.array([2, 1, 2])
y = np.array([1, 2, 2])

dot_product = 0
for i in range(len(x)):
    dot_product += x[i] * y[i]

print(dot_product)
