import numpy as np
import pandas as pd

df=pd.read_csv("/home/sumedha/Downloads/simulated_dataset.csv")
print(df.head())
print(df.describe())

x=df[["age","BMI","BP","blood_sugar","Gender"]]
y=df["disease_score_fluct"]
theta_0=0
theta_1=0
theta_2=0
theta_3=0
theta_4=0
x_bar=np.array(x)
y_bar=np.array(y)


for i in range ((len(x))):
    hypothesis=(theta_0*x_bar[i][0]+theta_1*x_bar[i][1]+theta_2*x_bar[i][2]+theta_3*x_bar[i][3]+theta_4*x_bar[i][4])


X=df[["age","BMI","BP","blood_sugar","Gender"]]
Y=df["disease_score_fluct"]

print(len(X))


# Cost function

def cost_function(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        alpha=0.01
        predictions = hypothesis(X, theta)
        error = predictions - y

        gradient = (1 / m) * np.dot(X.T, error)
        theta = theta - alpha * gradient

        cost = cost_function(X, y, theta)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i+1}, Cost = {cost}")

    return theta, cost_history


    print("\nFinal theta:", theta)

