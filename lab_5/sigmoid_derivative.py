import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

#sigmoid
y_sig = 1 / (1 + np.exp(-x))

#derivative
y_der = y_sig * (1 - y_sig)

#plot

plt.plot(x, y_der, label="derivative")

plt.legend()
plt.show()