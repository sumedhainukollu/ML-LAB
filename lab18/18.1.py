import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


data = [
    (6,5,'Blue'), (6,9,'Blue'), (8,6,'Red'), (8,8,'Red'),
    (8,10,'Red'), (9,2,'Blue'), (9,5,'Red'), (10,10,'Red'),
    (10,13,'Blue'), (11,5,'Red'), (11,8,'Red'), (12,6,'Red'),
    (12,11,'Blue'), (13,4,'Blue'), (14,8,'Blue')
]

X = np.array([[d[0], d[1]] for d in data])
y = np.array([1 if d[2]=='Red' else 0 for d in data])


def rbf_kernel(a, b, gamma=0.1):
    return np.exp(-gamma * np.sum((a - b)**2))


def polynomial_kernel(a, b):
    return (np.dot(a, b))**2


rbf_matrix = np.zeros((len(X), len(X)))
poly_matrix = np.zeros((len(X), len(X)))

for i in range(len(X)):
    for j in range(len(X)):
        rbf_matrix[i][j] = rbf_kernel(X[i], X[j])
        poly_matrix[i][j] = polynomial_kernel(X[i], X[j])


model_rbf = SVC(kernel='rbf', gamma=0.1)
model_poly = SVC(kernel='poly', degree=2)

model_rbf.fit(X, y)
model_poly.fit(X, y)


def plot_boundary(model, title):
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)

    colors = ['blue' if label==0 else 'red' for label in y]
    plt.scatter(X[:,0], X[:,1], c=colors)

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plot_boundary(model_rbf, "RBF Kernel")

plt.subplot(1,2,2)
plot_boundary(model_poly, "Polynomial Kernel")

plt.show()