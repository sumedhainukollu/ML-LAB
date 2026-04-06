from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv("/home/sumedha/Downloads/All_Datasets/Kaggle_datasets/Iris.csv")
data=data.dropna()
print(data.head())
x=data.drop("Outcome",axis=1)
le=LabelEncoder()
data["Outcome"]=le.fit_transform(data["Outcome"])
y=data["Outcome"]


def eda(x,y):
    #print("dimensions of data are",data.dimensions())
    #print(x.)
    print(y.shape)
    corr=data.corr()
    print(corr)
    print(data.isnull().sum())  #null values
    plt.figure(figsize=(10,8))
    sns.heatmap(corr,annot=True,cmap="coolwarm")
    plt.title("Correlation matrix")
    plt.figure(figsize=(12,6))
    sns.boxplot(data=data)
    plt.xticks(rotation=45)
    plt.title("Boxplot for Outlier Detection")
    plt.show()
    data.hist(figsize=(12,10))
    plt.show()
    return x,y
def scale_data(x,y):
    sc=StandardScaler()
    x_scaled=sc.fit_transform(x)
    return x_scaled
def split_data(x_scaled,y):
    x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)
    return x_train,x_test,y_train,y_test
def model(x_train,x_test,y_train,y_test):
    model=BaggingRegressor(estimator=DecisionTreeRegressor(),n_estimators=50,random_state=42)
    model.fit(x_train,y_train)
    ypred=model.predict(x_test)
    print("mean_square error is ",mean_squared_error(y_test,ypred),"r2 score is ",r2_score(y_test,ypred))
    return model,ypred
def main():
    eda(x,y)
    x_scaled=scale_data(x,y)
    x_train,x_test,y_train,y_test=split_data(x_scaled,y)
    model(x_train,x_test,y_train,y_test)
main()







