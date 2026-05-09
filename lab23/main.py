import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("Iris.csv")

# Use only required columns
df = df[['SepalLengthCm', 'SepalWidthCm', 'Species']]

# Add random noise
np.random.seed(42)
df['SepalLengthCm'] += np.random.normal(0, 0.2, size=len(df))
df['SepalWidthCm'] += np.random.normal(0, 0.2, size=len(df))

# Discretization (binning)
df['SepalLengthCm'] = pd.cut(df['SepalLengthCm'], bins=5, labels=False)
df['SepalWidthCm'] = pd.cut(df['SepalWidthCm'], bins=5, labels=False)

# Train-test split
train, test = train_test_split(df, test_size=0.3, random_state=42)



# Calculate P(Y)
P_Y = train['Species'].value_counts(normalize=True)

# Calculate P(X|Y)
# Create frequency tables
P_X_given_Y = {}

for cls in train['Species'].unique():
    subset = train[train['Species'] == cls]

    prob_table = subset.groupby(['SepalLengthCm', 'SepalWidthCm']).size()
    prob_table = prob_table / len(subset)

    P_X_given_Y[cls] = prob_table


# Prediction function
def predict(row):
    probs = {}

    for cls in P_Y.index:
        px_y = P_X_given_Y[cls].get((row['SepalLengthCm'], row['SepalWidthCm']), 1e-6)
        probs[cls] = P_Y[cls] * px_y

    return max(probs, key=probs.get)


# Predict
y_pred_gen = test.apply(predict, axis=1)
gen_acc = accuracy_score(test['Species'], y_pred_gen)

print("Generative Model Accuracy:", gen_acc)

# -------------------------------
# Decision Tree
# -------------------------------

X_train = train[['SepalLengthCm', 'SepalWidthCm']]
y_train = train['Species']

X_test = test[['SepalLengthCm', 'SepalWidthCm']]
y_test = test['Species']

dt = DecisionTreeClassifier(max_depth=2, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
dt_acc = accuracy_score(y_test, y_pred_dt)

print("Decision Tree Accuracy:", dt_acc)