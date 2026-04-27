import numpy as np
from sklearn.tree import DecisionTreeRegressor

class SimpleEnsemble:
    def __init__(self, n_estimators=5):
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            model = DecisionTreeRegressor()
            model.fit(X, y)
            self.models.append(model)

    def predict(self, X):
        preds = np.array([model.predict(X) for model in self.models])
        return np.mean(preds, axis=0)


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1)

    model = SimpleEnsemble(n_estimators=10)
    model.fit(X, y)
    preds = model.predict(X)
    print(preds[:5])