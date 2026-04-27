import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.preprocessing import LabelEncoder


# 🔹 Load data
def load_data():
    data = pd.read_csv("/home/sumedha/Downloads/All_Datasets/Kaggle_datasets/Iris.csv")
    data = data.dropna()

    X = data.drop("Species", axis=1).values
    y = data["Species"].values

    y = LabelEncoder().fit_transform(y)

    return X, y


# 🔹 Entropy
def entropy(y):
    counts = Counter(y)
    total = len(y)

    e = 0
    for c in counts.values():
        p = c / total
        e -= p * math.log2(p)

    return e


#
def best_split(X, y):
    best_gain = -1
    best_feature = 0
    best_thresh = 0

    for f in range(X.shape[1]):
        for t in X[:, f]:

            left = y[X[:, f] <= t]
            right = y[X[:, f] > t]

            if len(left) == 0 or len(right) == 0:
                continue

            p = len(y)
            gain = entropy(y) - (
                    (len(left) / p) * entropy(left) +
                    (len(right) / p) * entropy(right)
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_thresh = t

    return best_feature, best_thresh


# 🔹 Build tree (very simple)
def build_tree(X, y, depth=0, max_depth=3):
    # stop condition
    if len(set(y)) == 1 or depth == max_depth:
        return Counter(y).most_common(1)[0][0]

    f, t = best_split(X, y)

    left_idx = X[:, f] <= t
    right_idx = X[:, f] > t

    left = build_tree(X[left_idx], y[left_idx], depth + 1, max_depth)
    right = build_tree(X[right_idx], y[right_idx], depth + 1, max_depth)

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


# 🔹 Accuracy
def accuracy(y, y_pred):
    return np.mean(y == y_pred)


# 🔹 Main
def main():
    X, y = load_data()

    tree = build_tree(X, y)

    preds = predict_all(X, tree)

    print("Accuracy:", accuracy(y, preds))


main()