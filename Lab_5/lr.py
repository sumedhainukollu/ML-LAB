import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
#EDA
data=pd.read_csv("/home/sumedha/Downloads/All_Datasets/Kaggle_datasets/diabetes.csv")

print(data.columns)
x=data.drop("Outcome",axis=1)
y=data["Outcome"]
print(x.describe())
print(x.head())
print(x.info(),x.shape)
print(y.describe(),y.head(),y.info(),y.shape)

#train,test and split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#scale the data
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

#model selection,cross validation
lrmodel=LinearRegression()
lrscore=cross_val_score(lrmodel,x_train_scaled,y_train,cv=10)

#model fit
lrmodel.fit(x_train_scaled,y_train)

#y_pred
ypred=lrmodel.predict(x_test_scaled)

#r2score,mse
mse=mean_squared_error(y_test,ypred)
r2score=r2_score(y_test,ypred)

print(ypred,mse)