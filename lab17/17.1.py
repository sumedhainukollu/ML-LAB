import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.ma.core import product

# x1=[1,1,2,3,6,9,13,18,3,6,6,9,,10,11,12,16]
# x2=[13,18,9,6,3,2,1,1,15,6,11,5,10,5,10,5,6,3]

dict = [
    {"x1": 1, "x2": 13, "label": 1},
    {"x1": 1, "x2": 18, "label": 1},
    {"x1": 2, "x2": 9,  "label": 1},
    {"x1": 3, "x2": 6,  "label": 1},
    {"x1": 6, "x2": 3,  "label": 1},
    {"x1": 9, "x2": 2,  "label": 1},
    {"x1": 13, "x2": 1, "label": 1},
    {"x1": 18, "x2": 1, "label": 1},
    {"x1": 3, "x2": 15, "label": 0},
    {"x1": 6, "x2": 6,  "label": 0},
    {"x1": 6, "x2": 11, "label": 0},
    {"x1": 9, "x2": 5,  "label": 0},
    {"x1": 10, "x2": 10,"label": 0},
    {"x1": 11, "x2": 5, "label": 0},
    {"x1": 12, "x2": 6, "label": 0},
    {"x1": 16, "x2": 3, "label": 0},
]

data= pd.DataFrame(dict)
x1=np.array(data["x1"])
x2=np.array(data["x2"])
y=np.array(data["label"])

# 2D plot
for i in range(len(x1)):
    if y[i] == 1:
        plt.scatter(x1[i], x2[i], color='blue')
    else:
        plt.scatter(x1[i], x2[i], color='red')

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Original 2D Data")
plt.show()


# print(x1)
# plt.scatter(x1,x2)
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.show()


def transform(x1,x2):
    return x1 ** 2,math.sqrt(2)*x1*x2, x2**2


a,b,c=transform(x1,x2)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Scatter
for i in range(len(x1)):
    if y[i] == 1:
        ax.scatter(a[i], b[i], c[i], color='blue')
    else:
        ax.scatter(a[i], b[i], c[i], color='red')

# ax.scatter(a, b, c)

# Labels
ax.set_xlabel("A")
ax.set_ylabel("B")
ax.set_zlabel("C")

plt.show()



x=transform(3,6)
y=transform(10,10)

prod=np.dot(x,y)

print(prod)

x1 = [3, 6]
x2 = [10, 10]
def polynomial_kernel(x1,x2):
    return x1[0] ** 2 * x2[0] ** 2 + 2 * x1[0] * x2[0] * x1[1] * x2[1] + x1[1] ** 2 * x2[1] ** 2


print(polynomial_kernel(x1,x2))
