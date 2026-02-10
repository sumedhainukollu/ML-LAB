import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# Load data

import pandas as pd

# Load data from CSV
df = pd.read_csv("/home/sumedha/Downloads/simulated_dataset.csv")

# Features and target
x = df[["age", "BMI", "BP", "blood_sugar", "Gender"]].values
y = df["disease_score"].values


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)


# Feature scaling

x_mean = x_train.mean(axis=0)
x_std = x_train.std(axis=0)

x_train = (x_train - x_mean) / x_std
x_test  = (x_test  - x_mean) / x_std


# Convert to arrays

x_train = np.array(x_train)
x_test  = np.array(x_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)




# Initialize theta
theta_0=0
theta_1=theta_2=theta_3=theta_4=theta_5=0

lr = 0.0001
iterations = 1000


# Gradient Descent

for it in range(iterations):


    # Prediction

    y_pred = np.zeros(m)

    for i in range(m):
        y_pred[i] = theta_0
        for j in range(n):
            y_pred[i] = y_pred[i]+theta[j] * x_train[i][j]


    # Cost

    cost = 0
    for i in range(m):
        cost = cost+(y_pred[i] - y_train[i]) ** 2


    # Gradients

    grad_0 = 0
    grad = np.zeros(n)

    for i in range(m):
        error = y_pred[i] - y_train[i]
        grad_0 = grad_0+error
        for j in range(n):
            grad[j] = grad[j]+error * x_train[i][j]


    # Update

    theta_0 -= lr * grad_0
    for j in range(n):
        theta[j] = theta[j]-lr * grad[j]

    if it % 200 == 0:
        print("Iteration:", it, "Cost:", cost)


# Test set
y_test_pred = np.zeros(len(x_test))

for i in range(len(x_test)):
    y_test_pred[i] = theta_0      # ADD THIS
    for j in range(n):
        y_test_pred[i] = y_test_pred[i]+theta[j] * x_test[i][j]


print("\nFinal theta:")
print(theta)

print("\nR2 score (TEST):")
print(r2_score(y_test, y_test_pred))
