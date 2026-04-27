from random import shuffle

import numpy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score
from sklearn.preprocessing import StandardScaler
from statsmodels.sandbox.tools.cross_val import KFold
from sklearn.model_selection import KFold
#EDA
data=pd.read_csv("/home/sumedha/Downloads/All_Datasets/Kaggle_datasets/diabetes (1).csv")
print(data.columns)
x=data.drop("Outcome",axis=1)
y=data["Outcome"]
#print(x.describe())
#print(x.shape)
#print(x.info())

#print(y.describe())
#print(y.info())
#print(y.shape)


#train,test and split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#scale data
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)


#model_selection
lrmodel=LogisticRegression(max_iter=200)
kfold=KFold(n_splits=10,shuffle=True,random_state=42)
lrscores=cross_val_score(lrmodel,x_train_scaled,y_train,cv=kfold)
print(lrscores.mean(),lrscores.std())
#modelfit
lrmodel.fit(x_train_scaled,y_train)

#y_pred
ypred=lrmodel.predict(x_test_scaled)

#accuracy
accuracy=accuracy_score(y_test,ypred)
print("logistic regression accuracy is",accuracy)

#random forest
rfmodel=RandomForestClassifier(n_estimators=50)
rfscores=cross_val_score(rfmodel,x_train_scaled,y_train,cv=10)
print("mean of rf scores : ",rfscores.mean(),"mean of rf scored deviation is : ",rfscores.std())
rfmodel.fit(x_train_scaled,y_train)
ypred_rf=rfmodel.predict(x_test_scaled)

rfaccuracy=accuracy_score(y_test,ypred_rf)
print("random forest accuracy is:",rfaccuracy)