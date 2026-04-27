import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso,Ridge
from sklearn.metrics import accuracy_score

data=pd.read_csv("/home/sumedha/Downloads/All_Datasets/ISLP_Datasets/Publication.csv")
print("data.columns are :",data.columns)
print(data["mech"].unique())
data=pd.get_dummies(data,columns=["mech"],drop_first=True)
x=data.drop("status",axis=1)
y=data["status"]
print("x and y dimensions are ; ",x.shape,y.shape)

#split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#scale data
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

#model
lrmodel=LogisticRegression(max_iter=200)
lrmodel.fit(x_train_scaled,y_train)
ypred=lrmodel.predict(x_test_scaled)

acc=accuracy_score(y_test,ypred)
print("logistic reg accuracy ",acc)

#crossval
kfold=KFold(n_splits=10,shuffle=True,random_state=42)
cv=cross_val_score(lrmodel,x_train_scaled,y_train,cv=kfold)
print("mean and std of accuracy post cv are :",cv.mean(),cv.std())

#regularization
lasso=LogisticRegression(penalty="l1",solver="liblinear")
lasso.fit(x_train_scaled,y_train)
lpred=lasso.predict(x_test_scaled)
lassoacc=accuracy_score(y_test,lpred)
print("lassoacc are :",lassoacc)
#ridge
ridge=LogisticRegression(penalty="l2",solver="lbfgs",max_iter=100)
ridge.fit(x_train_scaled,y_train)
rpred=ridge.predict(x_test_scaled)
racc=accuracy_score(y_test,rpred)

print("ridge acc: ",rpred,racc)