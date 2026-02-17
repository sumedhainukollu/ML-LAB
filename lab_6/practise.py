import pandas as pd
import numpy as np

from sklearn.metrics import r2_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#import iris dataset
iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.features_name)

x=df.drop("target",axis=1)
y=df("target")



