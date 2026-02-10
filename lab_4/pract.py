import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data=fetch_california_housing()
x=data.data
y=data.target

print("input features",x.shape)
print("targets are",y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
print("train shape",x_train.shape)
print("test shape",x_test.shape)

x_train = np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)


#hypothesis
theta_0=0
theta_1=1
theta_2=0
theta_3=0
theta_4=0
theta_5=0


for j in range(1000):
    y_pred=[]
    for i in range((len(x_train))):
        hypothesis = (theta_0*x_train[1] + theta_2*x_train[2]+ theta_3*x_train[3]+theta_4*x_train[4]+ theta_5*x_train[5])
        y_pred.append(hypothesis)

    cost_fun=0
    for i in range(len(x_train)):
        cost_fun = cost_fun + ((y_pred[i] - y_train[i]) * (y_pred[i] - y_train[i]) )

        #gradient descent
lr=0.1
gd_0=0
gd_1=1
gd_2=2
gd_3=0
gd_4=0
gd_5=0
for k in range((len(x_train))):
    gd_0=((y_pred)-(y_train[0]))
    gd_1=((y_pred)-(y_train[1])*x_train[k][1])
    gd_2=((y_pred)-(y_train[2])*x_train[k][2])
    gd_3 = ((y_pred) - (y_train[3]) * x_train[k][3])
    gd_4 = ((y_pred) - (y_train[4]) * x_train[k][4])
    gd_5 = ((y_pred) - (y_train[5]) * x_train[k][5])
    theta_0 = theta_0 * lr * gd_0
    theta_1 = theta_1 * lr * gd_1
    theta_2= theta_2 * lr * gd_2
    theta_3 = theta_3 * lr * gd_3
    theta_4 = theta_4 * lr * gd_4
    theta_5 = theta_5 * lr * gd_5

    print(theta_0,theta_1,theta_2,theta_3,theta_4,theta_5)
    r2_score=r2_score(y_train,y_pred)







