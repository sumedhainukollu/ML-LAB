import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load data
df = pd.read_csv("/home/sumedha/Downloads/Iris.csv")

df = df[['SepalLengthCm', 'SepalWidthCm', 'Species']]

# Std
std_len = df['SepalLengthCm'].std()
std_width = df['SepalWidthCm'].std()

# Noise
noise_len = 0.1 * std_len
noise_width = 0.1 * std_width

df['SepalLengthCm'] += np.random.normal(0, noise_len, len(df))
df['SepalWidthCm'] += np.random.normal(0, noise_width, len(df))

# Discretize
df['SepalLengthCm_bins'] = pd.cut(df['SepalLengthCm'], bins=5, labels=False)
df['SepalWidthCm_bins'] = pd.cut(df['SepalWidthCm'], bins=5, labels=False)

# Split
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Define
target = 'Species'
features = ['SepalLengthCm_bins', 'SepalWidthCm_bins']

# Joint probability (correct)
prob_table = train_df.groupby(features + [target]).size()
joint_prob = prob_table / len(train_df)


# Prediction function
def predict_joint(row):
    probs = {}

    for cls in train_df[target].unique():
        key = (row['SepalLengthCm_bins'], row['SepalWidthCm_bins'], cls)
        probs[cls] = joint_prob.get(key, 1e-6)

    return max(probs, key=probs.get)


test_df['JP_pred'] = test_df.apply(predict_joint, axis=1)

#accuracy
jp_acc = accuracy_score(test_df[target], test_df['JP_pred'])
print("Joint Probability Accuracy:", jp_acc)

#dt

X_train = train_df[['SepalLengthCm', 'SepalWidthCm']]
y_train = train_df['Species']

X_test = test_df[['SepalLengthCm', 'SepalWidthCm']]
y_test = test_df['Species']

dt = DecisionTreeClassifier(max_depth=2)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

dt_acc = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", dt_acc)