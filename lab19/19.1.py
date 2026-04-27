import pandas as pd
import numpy as np

df = pd.read_csv("/home/sumedha/Downloads/All_Datasets/Kaggle_datasets/Heart.csv")
df = df.drop("Unnamed: 0", axis=1)
df = df.dropna()

# convert target
df["AHD"] = df["AHD"].map({"No": 0, "Yes": 1})

# convert categorical → numeric
df = pd.get_dummies(df, drop_first=True)

# split features/target
X = df.drop("AHD", axis=1)
y = df["AHD"]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# probabilities (IMPORTANT)
y_probs = model.predict_proba(X_test)[:, 1]
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# try different thresholds
thresholds = [0.3, 0.5, 0.7]

for t in thresholds:
    y_pred = (y_probs >= t).astype(int)

    TP = np.sum((y_test == 1) & (y_pred == 1))
    TN = np.sum((y_test == 0) & (y_pred == 0))
    FP = np.sum((y_test == 0) & (y_pred == 1))
    FN = np.sum((y_test == 1) & (y_pred == 0))

    accuracy = (TP + TN) / len(y_test)
    precision = TP / (TP + FP)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1 = 2 * precision * sensitivity / (precision + sensitivity)

    print(f"\nThreshold = {t}")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("F1-score:", f1)