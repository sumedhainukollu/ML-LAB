import math
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()
X = data.data   # All 8 features

rows = len(X)
cols = len(X[0])

# Create empty matrix for standardized values
standardized = [[0 for _ in range(cols)] for _ in range(rows)]

for j in range(cols):   # For each feature (column)

    # ---- Extract column ----
    column = []
    for i in range(rows):
        column.append(X[i][j])

    # ---- Compute Mean ----
    total = 0
    for val in column:
        total += val
    mean = total / rows

    # ---- Compute SD ----
    var_sum = 0
    for val in column:
        var_sum += (val - mean) ** 2

    variance = var_sum / rows
    sd = math.sqrt(variance)

    #standardize column
    for i in range(rows):
        standardized[i][j] = (X[i][j] - mean) / sd


#first 5 rows
print("First 5 rows after standardization:")
for i in range(5):
    print(standardized[i])


# verify
print("\nVerification (Mean ≈ 0, SD ≈ 1 for each feature):")

for j in range(cols):

    column = []
    for i in range(rows):
        column.append(standardized[i][j])

    mean = sum(column) / rows

    var_sum = 0
    for val in column:
        var_sum += (val - mean) ** 2

    sd = math.sqrt(var_sum / rows)

    print(f"Feature {j}: Mean = {mean:.5f}, SD = {sd:.5f}")