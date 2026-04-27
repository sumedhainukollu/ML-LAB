import math
from collections import Counter


# -----------------------
# Entropy (your version)
# -----------------------
def entropy(targets):
    tot = len(targets)
    count = Counter(targets)
    ent = 0.0

    for c in count.values():
        p = c / tot
        ent -= p * math.log2(p)

    return ent


# -----------------------
# Split dataset
# -----------------------
def split_data(features, targets, threshold):
    left_y, right_y = [], []
    left_X, right_X = [], []

    for i in range(len(features)):
        if features[i] <= threshold:
            left_X.append(features[i])
            left_y.append(targets[i])
        else:
            right_X.append(features[i])
            right_y.append(targets[i])

    return left_X, right_X, left_y, right_y


# -----------------------
# Best split (1 feature case)
# -----------------------
def best_split(features, targets):
    best_thresh = None
    best_gain = -1

    parent_entropy = entropy(targets)

    for thresh in features:
        left_X, right_X, left_y, right_y = split_data(features, targets, thresh)

        if len(left_y) == 0 or len(right_y) == 0:
            continue

        w_l = len(left_y) / len(targets)
        w_r = len(right_y) / len(targets)

        child_entropy = w_l * entropy(left_y) + w_r * entropy(right_y)

        gain = parent_entropy - child_entropy

        if gain > best_gain:
            best_gain = gain
            best_thresh = thresh

    return best_thresh


# -----------------------
# Build tree (very basic)
# -----------------------
def build_tree(features, targets):
    # If pure
    if len(set(targets)) == 1:
        return targets[0]

    thresh = best_split(features, targets)

    if thresh is None:
        return Counter(targets).most_common(1)[0][0]

    left_X, right_X, left_y, right_y = split_data(features, targets, thresh)

    return {
        "threshold": thresh,
        "left": build_tree(left_X, left_y),
        "right": build_tree(right_X, right_y)
    }


# -----------------------
# Prediction
# -----------------------
def predict(tree, x):
    if not isinstance(tree, dict):
        return tree

    if x <= tree["threshold"]:
        return predict(tree["left"], x)
    else:
        return predict(tree["right"], x)


# -----------------------
# Run example
# -----------------------
features = [1, 2, 3, 4]
targets = ["yes", "no", "yes", "no"]

tree = build_tree(features, targets)

print("Tree:", tree)

# Test prediction
for f in features:
    print(f, "->", predict(tree, f))