import pandas as pd
import numpy as np



df=pd.read_csv("/home/sumedha/Downloads/simulated_dataset.csv")
print(df.head())
print(df.describe())

X=df[["age","BMI","BP","blood_sugar","Gender"]]
Y=df["disease_score"]

print(len(X))

theta_0 = 0
theta_1 = 0
theta_2 = 0
theta_3 = 0
theta_4 = 0
X_bar = np.array(X)
y_bar = np.array(Y)

for j in range(50):

    print("after",j,"iteration")
    y_pred=[]

    for i in range(len(X)):
        Hypothesis=(theta_0*X_bar[i][0])+(theta_1*X_bar[i][1])+(theta_2*X_bar[i][2])+(theta_3*X_bar[i][3])+(theta_4*X_bar[i][4])
        y_pred.append(Hypothesis)

    cost_function=0
    for i in range(len(X)):
        cost_function=cost_function+((y_pred[i]-y_bar[i])*(y_pred[i]-y_bar[i]))

    print("cost function",cost_function)

    #update theta values
    for k in range(len(X)):
        gd0=((y_pred[i]-y_bar[i])*X_bar[i][0])
        gd1 =((y_pred[i] - y_bar[i]) * X_bar[1][1])
        gd2 =((y_pred[i] - y_bar[i]) * X_bar[i][2])
        gd3 =((y_pred[i] - y_bar[i]) * X_bar[i][3])
        gd4 =((y_pred[i] - y_bar[i]) * X_bar[i][4])

    theta_0=theta_0-0.00001*gd0
    theta_1=theta_1-0.00001*gd1
    theta_2=theta_2-0.00001*gd2
    theta_3=theta_3-0.00001*gd3
    theta_4=theta_4-0.00001*gd4

    print("updated t0=",theta_0)
    print("updated t1=",theta_1)
    print("updated t2=",theta_2)
    print("updated t3=",theta_3)
    print("updated t4=",theta_4)
