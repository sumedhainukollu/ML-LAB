from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv("/home/sumedha/Downloads/All_Datasets/Kaggle_datasets/diabetes (1).csv")
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
def scale_data(x):
    sc=StandardScaler()
    x_scaled=sc.fit_transform(x)
    return x_scaled
def model():
    basemodel=DecisionTreeRegressor(max_depth=4,min_samples_split=5,max_leaf_nodes=2)
    model=BaggingRegressor(estimator=basemodel,n_estimators=50,random_state=42)
    return model
def crossval(model,x_scaled,y):
    mo=model()
    scores_r2=cross_val_score(mo,x_scaled,y,cv=10,scoring='r2')
    scores_mse=cross_val_score(mo,x_scaled,y,cv=10,scoring='neg_mean_squared_error')
    print(scores_mse,scores_r2)
    return scores_mse,scores_r2
def hyperparameter(x_scaled,y):

    model=BaggingRegressor(estimator=DecisionTreeRegressor(),n_estimators=50,random_state=42)
    paramgrid={
        'n_estimators' : [10,50,100],
        'estimator__max_depth' :  [3,5,10],
        'estimator__min_samples_split': [2,5,10]
    }
    grid=GridSearchCV(model,paramgrid,cv=10,scoring='r2')
    grid.fit(x_scaled,y)
    print(grid.best_params_)
    return grid.best_estimator_
def main():
    eda(x,y)
    x_scaled=scale_data(x)
    model()
    scores_mse, scores_r2=crossval(model,x_scaled,y)
    hyperparameter(x_scaled,y)
main()
