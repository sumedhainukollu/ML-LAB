import numpy as np
import matplotlib.pyplot as plt

#formula
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#assign x values
x_values = np.linspace(-10, 10, 100)
y_values = sigmoid(x_values)

#graph
plt.plot(x_values, y_values)

# 4. Add labels so we know what we are looking at
plt.title("sigmoid Curve")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")

#show the plot
plt.show()