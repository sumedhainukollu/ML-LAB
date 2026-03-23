import numpy as np
from sklearn.datasets import load_diabetes


# 🔹 Load dataset
def load_data():
    data = load_diabetes()
    X = data.data
    y = data.target
    return X, y


# 🔹 Mean Squared Error
def mse(y):
    if len(y) == 0:
        return 0
    return np.mean((y - np.mean(y))**2)


# 🔹 Best split
def best_split(X, y):
    best_feature = 0
    best_thresh = 0
    best_error = float("inf")

    for f in range(X.shape[1]):
        for t in X[:, f]:

            left = y[X[:, f] <= t]
            right = y[X[:, f] > t]

            if len(left) == 0 or len(right) == 0:
                continue

            error = (len(left)*mse(left) + len(right)*mse(right)) / len(y)

            if error < best_error:
                best_error = error
                best_feature = f
                best_thresh = t

    return best_feature, best_thresh


# 🔹 Build tree
def build_tree(X, y, depth=0, max_depth=3):
    if depth == max_depth or len(y) < 5:
        return np.mean(y)

    f, t = best_split(X, y)

    left_idx = X[:, f] <= t
    right_idx = X[:, f] > t

    left = build_tree(X[left_idx], y[left_idx], depth+1, max_depth)
    right = build_tree(X[right_idx], y[right_idx], depth+1, max_depth)

    return (f, t, left, right)


# 🔹 Predict
def predict(x, tree):
    if not isinstance(tree, tuple):
        return tree

    f, t, left, right = tree

    if x[f] <= t:
        return predict(x, left)
    else:
        return predict(x, right)


def predict_all(X, tree):
    return np.array([predict(x, tree) for x in X])


# 🔹 Evaluation (MSE)
def evaluate(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


# 🔹 Main
def main():
    X, y = load_data()

    tree = build_tree(X, y)

    preds = predict_all(X, tree)

    print("MSE:", evaluate(y, preds))


main()