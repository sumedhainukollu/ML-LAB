import numpy as np

# Coefficient vector (theta)
theta = np.array([2, 3, 3])

# Input matrix X (5 samples)
X = np.array([
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]
])

# Empty list to store results
result = []

# Manual matrix multiplication using loops
for i in range(len(X)):
    y = 0
    for j in range(len(theta)):
        y = y + X[i][j] * theta[j]
    result.append(y)

print("Xθ", result)
