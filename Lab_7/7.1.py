import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

# Load data
data = pd.read_csv("sonar.csv", header=None)
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


encoder = LabelEncoder()
y = encoder.fit_transform(y)


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic_regression', LogisticRegression(max_iter=1000))
])


kfold = KFold(n_splits=10, shuffle=True, random_state=42)


scores = cross_val_score(pipeline, x, y, cv=kfold, scoring='accuracy')

# Corrected the print formatting and method calls
print(f"mean accuracy",scores.mean())
print(f"standard deviation:",scores.std())
print(f"cross validation accuracies:",scores)