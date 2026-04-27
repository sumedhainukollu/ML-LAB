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

# Encode labels (e.g., 'R'/'M' to 0/1)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Added StandardScaler to the Pipeline for better performance
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic_regression', LogisticRegression(max_iter=1000))
])

# Define cross-validation strategy
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Calculate scores
scores = cross_val_score(pipeline, x, y, cv=kfold, scoring='accuracy')

# Corrected the print formatting and method calls
print(f"Mean Accuracy: {scores.mean():.4f}")
print(f"Standard Deviation: {scores.std():.4f}")
print(f"Cross Validation Accuracies: {scores}")