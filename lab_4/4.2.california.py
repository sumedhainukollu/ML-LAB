
# IMPORT LIBRARIES

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing



df = pd.read_csv("/home/sumedha/Downloads/simulated_dataset.csv")

x = df[["age","BMI","BP","blood_sugar","Gender"]]
y = df["disease_score"]

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.30, random_state=42
)

x_train_bar = np.array(x_train)
x_test_bar  = np.array(x_test)
y_train_bar = np.array(y_train)
y_test_bar  = np.array(y_test)

scaler = StandardScaler()
X_train_bar_scaled = scaler.fit_transform(x_train_bar)
X_test_bar_scaled  = scaler.transform(x_test_bar)

theta_0 = theta_1 = theta_2 = theta_3 = theta_4 = theta_5 = 0
alpha = 0.00001
iterations = 10000
c_f = []

for j in range(iterations):

    y_pred = []
    for i in range(len(X_train_bar_scaled)):
        hyp = (
            theta_0 +theta_1 * X_train_bar_scaled[i][0] +theta_2 * X_train_bar_scaled[i][1] + theta_3 * X_train_bar_scaled[i][2] +theta_4 * X_train_bar_scaled[i][3] +theta_5 * X_train_bar_scaled[i][4] )
        y_pred.append(hyp)

    cost_function = 0
    for i in range(len(y_train_bar)):
        cost_function += (y_pred[i] - y_train_bar[i])**2
    c_f.append(cost_function)

    k = random.randint(0, len(X_train_bar_scaled)-1)
    error = y_pred[k] - y_train_bar[k]

    theta_0 -= alpha * error
    theta_1 -= alpha * error * x_train_bar_scaled[k][0]
    theta_2 -= alpha * error * x_train_bar_scaled[k][1]
    theta_3 -= alpha * error * x_train_bar_scaled[k][2]
    theta_4 -= alpha * error * x_train_bar_scaled[k][3]
    theta_5 -= alpha * error * x_train_bar_scaled[k][4]

plt.plot(c_f)
plt.title("Cost Function vs Iterations (Simulated Dataset)")
plt.xlabel("Iterations")
plt.ylabel("Cost Function")
plt.show()

y_pred_test = []
for i in range(len(X_test_bar_scaled)):
    Hypothesis_test = (
        theta_0 +
        theta_1 * X_test_bar_scaled[i][0] +
        theta_2 * X_test_bar_scaled[i][1] +
        theta_3 * X_test_bar_scaled[i][2] +
        theta_4 * X_test_bar_scaled[i][3] +
        theta_5 * X_test_bar_scaled[i][4]
    )
    y_pred_test.append(Hypothesis_test)

r2_m = r2_score(y_test_bar, y_pred_test)
print("R2 :", r2_m)

plt.scatter(y_test, y_pred_test)
plt.plot(y_test, y_test)
plt.title("Predvsactual")
plt.show()



lr_sim = LinearRegression()
lr_sim.fit(x_train, y_train)
r2_sklearn= lr_sim.score(x_test, y_test)

print("sklearn R2:", r2_sklearn)



cal = fetch_california_housing(as_frame=True)
df_cal = cal.frame

X = df_cal.drop("MedHouseVal", axis=1)
Y = df_cal["MedHouseVal"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.30, random_state=42
)

X_train_bar = np.array(X_train)
X_test_bar  = np.array(X_test)
Y_train_bar = np.array(Y_train)
Y_test_bar  = np.array(Y_test)

scaler = StandardScaler()
X_train_bar_scaled = scaler.fit_transform(X_train_bar)
X_test_bar_scaled  = scaler.transform(X_test_bar)

theta = np.zeros(X_train_bar_scaled.shape[1] + 1)
alpha = 0.001
iterations = 5000

for j in range(iterations):

    y_pred = []
    for i in range(len(X_train_bar_scaled)):
        Hypothesis = theta[0] + np.dot(theta[1:], X_train_bar_scaled[i])
        y_pred.append(Hypothesis)

    k = random.randint(0, len(X_train_bar_scaled)-1)
    error = y_pred[k] - Y_train_bar[k]

    theta[0] -= alpha * error
    for f in range(X_train_bar_scaled.shape[1]):
        theta[f+1] -= alpha * error * X_train_bar_scaled[k][f]

y_pred_test = []
for i in range(len(X_test_bar_scaled)):
    Hypothesis_test = theta[0] + np.dot(theta[1:], X_test_bar_scaled[i])
    y_pred_test.append(Hypothesis_test)

r2_manual_california = r2_score(Y_test_bar, y_pred_test)
print("Manual R2 (California Housing):", r2_manual_california)



lr_cal = LinearRegression()
lr_cal.fit(X_train, Y_train)
r2_sklearn_cal = lr_cal.score(X_test, Y_test)

print("Sklearn R2 (California Housing):", r2_sklearn_cal)

