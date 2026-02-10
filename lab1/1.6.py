import numpy as np

# Define inputs
X1 = np.linspace(-10, 10, 1)
X2 = np.linspace(10, 30, 1)
X3 = np.linspace(30, 50, 1)

gradient = []

# Compute y
y = (2*X1) + (3*X2) + (3*X3) + 4
print("y =", y)

# Partial derivative with respect to x1
dy_x1 = 2 * (X1**0)
gradient.append(dy_x1)

# Partial derivative with respect to x2
dy_x2 = 3 * (X2**0)
gradient.append(dy_x2)

# Partial derivative with respect to x3
dy_x3 = 3 * (X3**0)
gradient.append(dy_x3)

print("Gradient =", gradient)
