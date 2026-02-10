import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split



df=pd.read_csv("/home/sumedha/Downloads/simulated_dataset.csv")


X=df[["age","BMI","BP","blood_sugar","Gender"]]
Y=df["disease_score_fluct"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=999)

print(len(X_train))
print(len(X_test))
print(len(Y_test))
print(len(Y_train))

#design_matrix

X_train_bar=np.array(X_train)
X_test_bar=np.array(X_test)
Y_train_bar=np.array(Y_train)
Y_test_bar=np.array(Y_test)

#theta=np.array([[0],[0],[0],[0],[0]])
theta=np.dot(np.linalg.inv(np.dot(X_train_bar.T,X_train_bar)),(np.dot(X_train_bar.T,Y_train_bar)))
print(theta)


#define_hypothesis

y_pred_train=np.dot(X_train_bar,theta)




print(y_pred_train.shape)

#cost function


c_f=1/2*((np.dot((y_pred_train-Y_train_bar).T,y_pred_train-Y_train_bar)))

print(c_f)


y_pred_test=np.dot(X_test_bar,theta)


r2=r2_score(Y_test_bar,y_pred_test)

print(r2)

plt.scatter(Y_test,y_pred_test)
plt.plot(Y_test,Y_test)
plt.show()
