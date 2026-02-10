
# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset

df = pd.read_csv("/home/sumedha/Downloads/simulated_dataset.csv")
print(df.head())


# 2. Features and target

x = df[["age","BMI","BP","blood_sugar","Gender"]].values
y = df["disease_score_fluct"].values

print("X shape:", x.shape)
print("Y shape:", y.shape)


# 4. Normal Equation
#design matrix
xbar=x.transpose()

print(xbar)

#cost function

xr=xbar @ x
x_inv = np.linalg.inv(xr)

theta=x_inv @ xbar @ y
m = len(y)

cost_fun = (1 / 2 * (x@theta - y).T @  (x @ theta - y))

#cf=1/2*np.sum((y_pred-y)**2)
print("Cost is :", cost_fun)

print("thetas are",theta)


