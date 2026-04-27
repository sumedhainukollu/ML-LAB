import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------
# Step 1: Load Your Dataset
# ----------------------------
df = pd.read_csv("your_dataset.csv")

# ----------------------------
# Step 2: Separate Features and Target
# ----------------------------
# Replace 'Target' with your actual target column name
X = df.drop("Target", axis=1)
y = df["Target"]

# ----------------------------
# Step 3: Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ----------------------------
# Step 4: Train Decision Tree
# ----------------------------
model = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# Step 5: Predict
# ----------------------------
y_pred = model.predict(X_test)

# ----------------------------
# Step 6: Evaluate
# ----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", round(mse, 4))
print("R2 Score:", round(r2, 4))