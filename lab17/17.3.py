import numpy as np


def polynomial_kernel(a, b):
    return (a[0]**2 * b[0]**2 +
            2*a[0]*b[0]*a[1]*b[1] +
            a[1]**2 * b[1]**2)


a = np.array([3,6])
b = np.array([10,10])

kernel_value = polynomial_kernel(a, b)

print("Polynomial kernel result:", kernel_value)