import math
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()


X = data.data[:, 0]
lst = list(X)


def standardization(lst):

    l = len(lst)

    #mean
    total = 0
    for i in lst:
        total += i
    mean = total / l

    # variance
    var_sum = 0
    for i in lst:
        var_sum += (i - mean) ** 2

    variance = var_sum / l
    sd = math.sqrt(variance)

    #z-score transformation
    std_vals = []
    for i in lst:
        z = (i - mean) / sd
        std_vals.append(z)

    return std_vals


#apply standardization

standardized_values = standardization(lst)

print("First 10 standardized values:")
print(standardized_values[:10])


#verify Mean ≈ 0

new_mean = sum(standardized_values) / len(standardized_values)

#verify SD ≈ 1

var_sum = 0
for i in standardized_values:
    var_sum += (i - new_mean) ** 2

new_variance = var_sum / len(standardized_values)
new_sd = math.sqrt(new_variance)

print("\nMean after standardization:", new_mean)
print("SD after standardization:", new_sd)