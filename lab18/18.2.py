import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

mask = y != 0
X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

models = {
    "Linear": SVC(kernel="linear"),
    "RBF": SVC(kernel="rbf", gamma=0.5),
    "Polynomial": SVC(kernel="poly", degree=3)
}

plt.figure(figsize=(15,4))

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

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

    colors = ['red' if label==1 else 'blue' for label in y]
    plt.scatter(X[:,0], X[:,1], c=colors)

    plt.title(f"{name} Kernel\nAccuracy={acc:.2f}")

plt.show()