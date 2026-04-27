import numpy as np
def polynomial_kernel(a, b):
    return (a[0]**2 * b[0]**2 +
            2*a[0]*b[0]*a[1]*b[1] +
            a[1]**2 * b[1]**2)

a = np.array([3,6])
b = np.array([10,10])

phi_a = Transform(a[0], a[1])
phi_b = Transform(b[0], b[1])

dot_product = np.dot(phi_a, phi_b)
kernel_value = polynomial_kernel(a, b)

print("Dot product (transformed):", dot_product)
print("Polynomial kernel:", kernel_value)