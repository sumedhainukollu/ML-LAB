import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])

result = np.dot(A.T, A)
print("A transpose A:\n", result)