import numpy as np
import matplotlib.pyplot as plt

X = np.random.randint(-100, 100, 100)


print(X)
Y=[]
for i in X:
    y=2*i+3
    Y.append(y)

print(Y)

plt.plot(X,Y)

plt.title("y=2i+3")
plt.show()