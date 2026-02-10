import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



df=pd.read_csv("/home/sumedha/Downloads/simulated_dataset.csv")
print(df.head())
print(df.describe())

X=df[["age","BMI","BP","blood_sugar","Gender"]]
Y=df["disease_score"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=42)

print(len(X_train))
print(len(X_test))
print(len(Y_test))
print(len(Y_train))

theta_0 = 0
theta_1 = 0
theta_2 = 0
theta_3 = 0
theta_4 = 0
theta_5=0
X_test_bar = np.array(X_test)
Y_test_bar = np.array(Y_test)
X_train_bar = np.array(X_train)
Y_train_bar = np.array(Y_train)


scalar=StandardScaler()
scalar.fit(X_train_bar)
X_train_bar_scaled=scalar.transform(X_train_bar)
X_test_bar_scaled=scalar.transform(X_test_bar)


print(X_test_bar)

c_f=[]
for j in range(10,000):

     print("after",j,"iteration")
     y_pred=[]

     for i in range(len(X_train_bar_scaled)):
         Hypothesis=+theta_0+(theta_1*X_train_bar_scaled[i][0])+(theta_2*X_train_bar_scaled[i][1])+(theta_3*X_train_bar_scaled[i][2])+(theta_4*X_train_bar_scaled[i][3])+(theta_5*X_train_bar_scaled[i][4])
         y_pred.append(Hypothesis)

     cost_function=0
     for i in range(len(X_train_bar_scaled)):
         cost_function=cost_function+((y_pred[i]-Y_train_bar[i])*(y_pred[i]-Y_train_bar[i]))

     print("cost function",cost_function)
     c_f.append(cost_function)
     #update theta values
     gd0 = gd1 = gd2 = gd3 = gd4 = gd5 = 0

     k=random.randint(1,40)
     error = y_pred[k] - Y_train_bar[k]
     gd0 = gd0+error
     gd1 = gd1+error * X_train_bar_scaled[k][0]
     gd2 = gd2+error * X_train_bar_scaled[k][1]
     gd3 = gd3+error * X_train_bar_scaled[k][2]
     gd4 = gd4+error * X_train_bar_scaled[k][3]
     gd5 = gd5+error * X_train_bar_scaled[k][4]


     alpha = 0.001

     theta_0 -= alpha * (gd0)
     theta_1 -= alpha * (gd1)
     theta_2 -= alpha * (gd2)
     theta_3 -= alpha * (gd3)
     theta_4 -= alpha * (gd4)
     theta_5 -= alpha * (gd5)
#
#
     print("updated t0=", theta_0)
     print("updated t1=",theta_1)
     print("updated t2=",theta_2)
     print("updated t3=",theta_3)
     print("updated t4=",theta_4)
     print("updated t5=", theta_5)
#


#
#
plt.plot(c_f)
plt.show()
print("final updated t0=", theta_0)
print("final updated t1=", theta_1)
print("final updated t2=", theta_2)
print("final updated t3=", theta_3)
print("final updated t4=", theta_4)
print("final updated t5=", theta_5)

y_pred_test=[]
for i in range(len(X_test_bar_scaled)):
    Hypothesis_test = +theta_0+(theta_1 * X_test_bar_scaled[i][0]) + (theta_2 * X_test_bar_scaled[i][1]) + (theta_3 * X_test_bar_scaled[i][2]) + (theta_4 * X_test_bar_scaled[i][3]) + (theta_5 * X_test_bar_scaled[i][4])
    y_pred_test.append(Hypothesis_test)

cost_fun=0
for i in range(len(X_test_bar_scaled)):
    cost_fun=cost_fun+((y_pred_test[i]-Y_test_bar[i])*(y_pred_test[i]-Y_test_bar[i]))

print("cost_fun_in_test_data=", cost_function)


r2=r2_score(Y_test_bar,y_pred_test)

print(r2)


plt.scatter(Y_test,y_pred_test)
plt.plot(Y_test,Y_test)
plt.show()