import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Transform(x1, x2):
    return np.array([x1**2, np.sqrt(2)*x1*x2, x2**2])

data = [
    (1,13,'Blue'), (1,18,'Blue'), (2,9,'Blue'), (3,6,'Blue'),
    (6,3,'Blue'), (9,2,'Blue'), (13,1,'Blue'), (18,1,'Blue'),
    (3,15,'Red'), (6,6,'Red'), (6,11,'Red'), (9,5,'Red'),
    (10,10,'Red'), (11,5,'Red'), (12,6,'Red'), (16,3,'Red')
]

X = np.array([[d[0], d[1]] for d in data])
y = np.array([1 if d[2]=='Red' else 0 for d in data])

colors = ['blue' if label==0 else 'red' for label in y]

plt.scatter(X[:,0], X[:,1], c=colors)
plt.title("2D Data")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

X_trans = np.array([Transform(x[0], x[1]) for x in X])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_trans[:,0], X_trans[:,1], X_trans[:,2], c=colors)

xx, yy = np.meshgrid(
    np.linspace(X_trans[:,0].min(), X_trans[:,0].max(), 10),
    np.linspace(X_trans[:,1].min(), X_trans[:,1].max(), 10)
)

zz = -(xx + yy) + np.mean(X_trans[:,2])
ax.plot_surface(xx, yy, zz, alpha=0.3)

ax.set_xlabel("x1^2")
ax.set_ylabel("sqrt(2)x1x2")
ax.set_zlabel("x2^2")

plt.show()
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
from sklearn.svm import SVC

models = {
    "Linear": SVC(kernel='linear'),
    "Polynomial": SVC(kernel='poly', degree=2),
    "RBF": SVC(kernel='rbf', gamma=0.1)
}

plt.figure(figsize=(15,4))

for i, (name, model) in enumerate(models.items()):
    model.fit(X, y)

    plt.subplot(1,3,i+1)

    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=colors)

    plt.title(name + " Kernel")
    plt.xlabel("x1")
    plt.ylabel("x2")

plt.show()
