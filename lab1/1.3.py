import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-10, 10, 100)


print(X)
Y=[]
for i in X:
    y=(2*i*i)+(3*i)+4
    Y.append(y)

print(Y)

plt.plot(X,Y)
plt.xlabel("x_axis")
plt.ylabel("y_axis")
plt.title("y=2x2+3x+4")
plt.show()