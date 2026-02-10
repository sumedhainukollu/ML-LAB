import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
california = fetch_california_housing()

X = california.data
y = california.target

print("X shape:", X.shape)
print("y shape:", y.shape)



df = pd.DataFrame(X, columns=california.feature_names)
df["MedHouseValue"] = y
print(df.head())
print(df.describe())



X = (X - X.mean(axis=0)) / X.std(axis=0)


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# Add Bias Term

m_train = len(y_train)
m_test = len(y_test)

X_train = np.c_[np.ones(m_train), X_train]
X_test = np.c_[np.ones(m_test), X_test]


# Initialize Theta

theta = np.zeros(X_train.shape[1])


# Get dimensions

m, n = X.shape


# Scratch Linear Regression
theta = np.zeros(n)

alpha = 0.001
iterations = 5000

for _ in range(iterations):
    y_pred = X @ theta
    error = y_pred - y
    gradient = (1 / m) * (X.T @ error)
    theta = theta - alpha * gradient

print("\nTheta (Scratch, no bias):")
print(theta)




sk_model = LinearRegression(fit_intercept=False)
sk_model.fit(X, y)

# Predictions using sklearn
y_pred_sk = sk_model.predict(X)


r2 = r2_score(y, y_pred_sk)

print("\ntheta (Sklearn, ):")
print(sk_model.coef_)

print("\nR2 score (Sklearn):")
print(r2)
