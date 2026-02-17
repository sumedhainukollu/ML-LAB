import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.metrics import r2_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#load iris data
iris=load_iris()


df=pd.DataFrame(iris.data,columns=iris.feature_names)

df["target"] = iris.target

print(df.describe())
print(df.head())


x=df.drop("target",axis=1)
y=df.target


print(x)
print(y)



k=10
l=len(x)


df=df.sample(frac=1, random_state=42).reset_index(drop=True)


df["fold"] = np.random.randint(0, k, size=len(df))
print("fold",df["fold"])

print("x shape:", x.shape)

scores=[]

for i in range(k):
    df_test = df[df["fold"] == i]
    df_train = df[df["fold"] != i]
    x_train= df_train.drop(["target", "fold"], axis=1)
    x_test=df_test.drop(["target", "fold"], axis=1)
    y_train=df_train["target"]
    y_test=df_test["target"]
    print(f"{i}: Train: {x_train.shape}"
          f"{i}: Test: {x_test.shape}")
    model=LinearRegression()
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    score = r2_score(y_test, y_pred)
    scores.append(score)
print(scores)