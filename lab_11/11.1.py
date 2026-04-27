from collections import Counter
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
def load_data():
    data=pd.read_csv("/home/sumedha/Downloads/All_Datasets/Kaggle_datasets/Iris.csv")
    data=data.dropna()
    print(data.head())
    x=data.drop("Species",axis=1).values
    y=data["Species"].values
    enc=LabelEncoder()
    y=enc.fit_transform(y)
    return x,y
def split_data(x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    return x_train,x_test,y_train,y_test
def treemodel(x_train,y_train,x_test,y_test):
    tree=DecisionTreeClassifier()
    tree.fit(x_train,y_train)
    ypred=tree.predict(x_test)
    scores=accuracy_score(y_test,ypred)
    print("score is : ",scores)
    cv = cross_val_score(tree,x_train,y_train, cv=10)
    print(cv)
    return scores,ypred
def main():
    x,y=load_data()
    x_train,x_test,y_train,y_test=split_data(x,y)
    treemodel(x_train,y_train,x_test,y_test)
main()




