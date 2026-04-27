import numpy
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#data load
data=pd.read_csv("/home/sumedha/Downloads/All_Datasets/Kaggle_datasets/Iris.csv")
data=data.dropna()


#encode
encoder=LabelEncoder()
data["Species"]=encoder.fit_transform(data["Species"])
x=data.drop("Species",axis=1)
y=data["Species"]
print(x.shape,y.shape)


#split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)



#model
model=DecisionTreeClassifier(criterion="gini",max_depth=4,random_state=42)

model.fit(x_train,y_train)

ypred=model.predict(x_test)

acc=accuracy_score(y_test,ypred)
print(acc)
#cross val
kfold=KFold(n_splits=10,shuffle=True,random_state=42)
cvscores=cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')
print(cvscores.mean(),cvscores.std())