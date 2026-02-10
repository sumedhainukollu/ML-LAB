import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
# Load sklearn dataset

data = fetch_california_housing()

X = data.data          # (m, n)
y = data.target        # (m,)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Feature scaling

X = (X - X.mean(axis=0)) / X.std(axis=0)

# Get dimensions
#
m, n = X.shape


# Scratch Linear Regression

theta = np.zeros(n)

alpha = 0.01
iterations = 1000

for _ in range(iterations):
    y_pred = X @ theta
    error = y_pred - y
    gradient = (1 / m) * (X.T @ error)
    theta = theta - alpha * gradient

print("\nTheta (Scratch, ):")
print(theta)



# Scikit-learn Linear Regression

sk_model = LinearRegression(fit_intercept=False)
sk_model.fit(X, y)

theta_sklearn = sk_model.coef_

print("\nTheta (Sklearn, ):")
print(theta_sklearn)




# Compare theta values

print("\nDifference (Scratch − Sklearn):")
print(theta - theta_sklearn)


