import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

data=pd.read_csv("/home/sumedha/Downloads/All_Datasets/ISLP_Datasets/Advertising.csv")
print("columns are: ",data.columns)
x=data.drop("sales",axis=1)
y=data["sales"]
print("dimensions of x are : ",x.shape)
print("dimesions of y are : ",y.shape)

print("description of x is : ",x.describe())
print("description of y is ",y.describe())


