from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np




[X, Y] = fetch_california_housing(return_X_y=True)
print(X.shape)
print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=999)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)



theta_0 = 0
theta_1 = 0
theta_2 = 0
theta_3 = 0
theta_4 = 0
theta_5=0
theta_6=0
theta_7=0
theta_8=0



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
for j in range(500):

     print("after",j,"iteration")
     y_pred=[]

     for i in range(len(X_train_bar_scaled)):
         Hypothesis=+theta_0+(theta_1*X_train_bar_scaled[i][0])+(theta_2*X_train_bar_scaled[i][1])+(theta_3*X_train_bar_scaled[i][2])+(theta_4*X_train_bar_scaled[i][3])+(theta_5*X_train_bar_scaled[i][4])+(theta_6*X_train_bar_scaled[i][5])+(theta_7*X_train_bar_scaled[i][6])+(theta_8*X_train_bar_scaled[i][7])
         y_pred.append(Hypothesis)

     cost_function=0
     for i in range(len(X_train_bar_scaled)):
         cost_function=cost_function+((y_pred[i]-Y_train_bar[i])*(y_pred[i]-Y_train_bar[i]))

     print("cost function",cost_function)
     c_f.append(cost_function)
     #update theta values
     gd0 = gd1 = gd2 = gd3 = gd4 = gd5 = gd6 = gd7 = gd8 = 0

     for k in range(len(X_train_bar_scaled)):
          error = y_pred[k] - Y_train_bar[k]
          gd0 += error
          gd1 += error * X_train_bar_scaled[k][0]
          gd2 += error * X_train_bar_scaled[k][1]
          gd3 += error * X_train_bar_scaled[k][2]
          gd4 += error * X_train_bar_scaled[k][3]
          gd5 += error * X_train_bar_scaled[k][4]
          gd6 += error * X_train_bar_scaled[k][5]
          gd7 += error * X_train_bar_scaled[k][6]
          gd8 += error * X_train_bar_scaled[k][7]


     alpha = 0.00001

     theta_0 -= alpha * (gd0)
     theta_1 -= alpha * (gd1)
     theta_2 -= alpha * (gd2)
     theta_3 -= alpha * (gd3)
     theta_4 -= alpha * (gd4)
     theta_5 -= alpha * (gd5)
     theta_6 -= alpha * (gd6)
     theta_7 -= alpha * (gd7)
     theta_8 -= alpha * (gd8)

     print("updated t0=", theta_0)
     print("updated t1=",theta_1)
     print("updated t2=",theta_2)
     print("updated t3=",theta_3)
     print("updated t4=",theta_4)
     print("updated t5=", theta_5)
     print("updated t5=", theta_6)
     print("updated t5=", theta_7)
     print("updated t5=", theta_8)

plt.plot(c_f)
plt.show()
print("final updated t0=", theta_0)
print("final updated t1=", theta_1)
print("final updated t2=", theta_2)
print("final updated t3=", theta_3)
print("final updated t4=", theta_4)
print("final updated t5=", theta_5)
print("final updated t5=", theta_6)
print("final updated t5=", theta_7)
print("final updated t5=", theta_8)

y_pred_test=[]
for i in range(len(X_test_bar_scaled)):
    Hypothesis_test = theta_0+(theta_1 * X_test_bar_scaled[i][0]) + (theta_2 * X_test_bar_scaled[i][1]) + (theta_3 * X_test_bar_scaled[i][2]) + (theta_4 * X_test_bar_scaled[i][3]) + (theta_5 * X_test_bar_scaled[i][4])+(theta_6 * X_test_bar_scaled[i][5])+(theta_7 * X_test_bar_scaled[i][6])+(theta_8 * X_test_bar_scaled[i][7])
    y_pred_test.append(Hypothesis_test)

cost_fun=0
for i in range(len(X_test_bar_scaled)):
    cost_fun=cost_fun+((y_pred_test[i]-Y_test_bar[i])*(y_pred_test[i]-Y_test_bar[i]))

print("cost_fun_in_test_data=", cost_function)


r2=r2_score(Y_test_bar,y_pred_test)

print(r2)

plt.scatter(Y_test,y_pred)
plt.show()