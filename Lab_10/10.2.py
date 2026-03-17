import math
from collections import Counter

# Entropy function (reuse from Lab 10.1)
def entropy(labels):
    total = len(labels)
    count = Counter(labels)

    entropy_value = 0.0
    for c in count.values():
        p = c / total
        entropy_value -= p * math.log2(p)

    return entropy_value
# Information Gain function
def information_gain(parent_labels, left_labels, right_labels):
    total = len(parent_labels)

    # Entropy before split
    parent_entropy = entropy(parent_labels)

    # Weights
    left_weight = len(left_labels) / total
    right_weight = len(right_labels) / total

    # Entropy after split
    weighted_entropy = (
        left_weight * entropy(left_labels) +
        right_weight * entropy(right_labels)
    )

    # Information Gain
    ig = parent_entropy - weighted_entropy

    return ig

# Sample data
parent_labels = ["yes", "yes", "no", "no"]
left_labels = ["yes", "yes"]
right_labels = ["no", "no"]

# Output
print("Information Gain:", information_gain(parent_labels, left_labels, right_labels))