import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

#EDA
data=pd.read_csv("/home/sumedha/Downloads/All_Datasets/ISLP_Datasets/Boston.csv")
x=data.drop("medv",axis=1)
y=data["medv"]
print("data columns are :",data.columns)
print("dimensions of x and y are : ",x.shape,"\t",y.shape)
print(data.describe())
print(data.info())

#train,test and split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#scale data
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

#model linear regression
lrmodel=LinearRegression()
lrmodel.fit(x_train_scaled,y_train)
ypred=lrmodel.predict(x_test_scaled)
r2score=r2_score(y_test,ypred)
mse=mean_squared_error(y_test,ypred)

print("r2score of linear reg : ",r2score,"mse of linear reg : ",mse)

#cross validation
cross_val_scores=cross_val_score(lrmodel,x_train_scaled,y_train,cv=10,scoring='r2')
print("mean is cv : ",cross_val_scores.mean(),"std of cv is : ",cross_val_scores.std())

#regularization
lasso=Lasso(alpha=1.0)
lasso.fit(x_train_scaled,y_train)
lassopred=lasso.predict(x_test_scaled)
lassomse=mean_squared_error(y_test,lassopred)
lassor2=r2_score(y_test,lassopred)

print("lasso mse:",lassomse,"lassor2: ",lassor2)

ridge=Ridge(alpha=1.0)
ridge.fit(x_train_scaled,y_train)
ridgepred=ridge.predict(x_test_scaled)

ridgemse=mean_squared_error(y_test,ridgepred)
ridger2=r2_score(y_test,ridgepred)

print("ridge mse and r2score :",ridgemse,ridger2)

