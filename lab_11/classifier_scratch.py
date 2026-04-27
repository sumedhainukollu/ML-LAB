import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Entropy
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    prob = counts / counts.sum()
    return -np.sum(prob * np.log2(prob + 1e-9))

# Best split
def best_split(X, y):
    best_feature, best_thresh = None, None
    best_gain = -1
    parent_entropy = entropy(y)

    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])

        for thresh in thresholds:
            left_mask = X[:, feature] <= thresh
            right_mask = X[:, feature] > thresh

            if sum(left_mask) == 0 or sum(right_mask) == 0:
                continue

            y_l, y_r = y[left_mask], y[right_mask]

            w_l = len(y_l) / len(y)
            w_r = len(y_r) / len(y)

            child_entropy = w_l * entropy(y_l) + w_r * entropy(y_r)
            gain = parent_entropy - child_entropy

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_thresh = thresh

    return best_feature, best_thresh

# Build tree
def build_tree_classification(X, y, depth=0, max_depth=5):
    if depth >= max_depth or len(np.unique(y)) == 1:
        return np.bincount(y).argmax()

    feature, thresh = best_split(X, y)

    if feature is None:
        return np.bincount(y).argmax()

    left_mask = X[:, feature] <= thresh
    right_mask = X[:, feature] > thresh

    return {
        "feature": feature,
        "threshold": thresh,
        "left": build_tree_classification(X[left_mask], y[left_mask], depth+1, max_depth),
        "right": build_tree_classification(X[right_mask], y[right_mask], depth+1, max_depth)
    }

# Predict one
def predict_one_classification(tree, x):
    if not isinstance(tree, dict):
        return tree

    if x[tree["feature"]] <= tree["threshold"]:
        return predict_one_classification(tree["left"], x)
    else:
        return predict_one_classification(tree["right"], x)

# Predict all
def predict_classification(tree, X):
    return np.array([predict_one_classification(tree, x) for x in X])


# Run
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tree = build_tree_classification(X_train, y_train)

preds = predict_classification(tree, X_test)

print("Accuracy:", np.mean(preds == y_test))