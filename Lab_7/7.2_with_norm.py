from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


data = pd.read_csv("sonar.csv", header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


encoder = LabelEncoder()
y = encoder.fit_transform(y)


pipeline_scale = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])


kfold = KFold(n_splits=10, shuffle=True, random_state=42)

scores_scale = cross_val_score(pipeline_scale, X, y,
                               cv=kfold, scoring='accuracy')

print("\nwith normalization")
print("Mean Accuracy:", scores_scale.mean())
print("Std Dev:", scores_scale.std())
print("Fold Accuracies:", scores_scale)