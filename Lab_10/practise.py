import math
from collections import Counter


def entropy(labels):
    tot = len(labels)
    count = Counter(labels)

    entropy_value = 0.0
    for c in count.values():
        p = c / tot
        entropy_value -= p * math.log2(p)

    return entropy_value


def information_gain(parent_labels, left_labels, right_labels):
    tot = len(parent_labels)

    parent_ent = entropy(parent_labels)

    left_weight = len(left_labels) / tot
    right_weight = len(right_labels) / tot

    weighted_ent = (
            left_weight * entropy(left_labels) +
            right_weight * entropy(right_labels)
    )

    ig = parent_ent - weighted_ent
    return ig


# Data
parent_labels = ["yes", "yes", "no", "no"]
left_labels = ["yes", "yes"]
right_labels = ["no", "no"]

print("Information Gain:", information_gain(parent_labels, left_labels, right_labels))