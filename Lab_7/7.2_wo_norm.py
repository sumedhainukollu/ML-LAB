import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Load data
data = pd.read_csv("sonar.csv", header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


encoder = LabelEncoder()
y = encoder.fit_transform(y)


pipeline = Pipeline([
    ('logreg', LogisticRegression(max_iter=1000))
])

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='accuracy')

print("Without  Normalization")
print("Mean Accuracy:", scores.mean())
print("Std Dev:", scores.std())
print("Fold Accuracies:", scores)