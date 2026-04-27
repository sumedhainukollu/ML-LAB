import numpy as np
from sklearn.tree import DecisionTreeRegressor
from ISLP import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class SimpleGradientBooster:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):

        self.initial_prediction = np.mean(y)
        prediction = np.full(len(y), self.initial_prediction)

        for i in range(self.n_estimators):

            residuals = y - prediction


            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)


            prediction += self.learning_rate * tree.predict(X)

            # Save the tree
            self.trees.append(tree)

    def predict(self, X):

        prediction = np.full(len(X), self.initial_prediction)


        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(X)

        return prediction


boston = load_data('Boston')
X = boston.drop('medv', axis=1)
y = boston['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train our custom model
model = SimpleGradientBooster(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)
print(f"Scratch GBM RMSE: {mean_squared_error(y_test, preds, squared=False):.4f}")