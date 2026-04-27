import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.preprocessing import LabelEncoder


# 🔹 Load data
def load_data():
    data = pd.read_csv("/home/sumedha/Downloads/All_Datasets/Kaggle_datasets/Iris.csv")
    data = data.dropna()

    X = data.drop("Species", axis=1).values
    y = data["Species"].values

    y = LabelEncoder().fit_transform(y)

    return X, y

def entropy(y):
    
