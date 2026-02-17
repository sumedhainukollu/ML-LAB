import pandas as pd
import numpy as np

from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# -----------------------------
# Load California Housing data
# -----------------------------
housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["target"] = housing.target

print("Shape:", df.shape)
print(df.head())

# -----------------------------
# K-Fold Setup
# -----------------------------
k = 5

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Assign random fold labels
df["fold"] = np.random.randint(0, k, size=len(df))

scores = []

# -----------------------------
# K-Fold Loop
# -----------------------------
for i in range(k):

    df_test = df[df["fold"] == i]
    df_train = df[df["fold"] != i]

    X_train = df_train.drop(["target", "fold"], axis=1)
    y_train = df_train["target"]

    X_test = df_test.drop(["target", "fold"], axis=1)
    y_test = df_test["target"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    score = r2_score(y_test, y_pred)
    scores.append(score)

    print("Fold", i+1, "R2:", score)

print("\nAll Fold Scores:", scores)
print("Average R2:", np.mean(scores))
