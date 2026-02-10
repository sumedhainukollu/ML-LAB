import numpy as np
import pandas as pd

df = pd.read_csv("/home/sumedha/Downloads/simulated_dataset.csv")

X = df[["age","BMI","BP","blood_sugar","Gender"]].values
y = df["disease_score_fluct"].values

m = len(y)
X = np.c_[np.ones(m), X]
theta = np.zeros(X.shape[1])
def hypothesis(X, theta):
    return np.dot(X, theta)
def cost_function(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = hypothesis(X, theta)
        error = predictions - y

        gradient = (1 / m) * np.dot(X.T, error)
        theta = theta - alpha * gradient

        cost = cost_function(X, y, theta)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}, Cost = {cost}")

    return theta, cost_history
alpha = 0.000001
iterations = 2000

theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)

print("\nFinal theta:", theta)
