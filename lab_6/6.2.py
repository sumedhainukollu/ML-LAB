import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

data = fetch_california_housing()
X, y = data.data, data.target

X_train_full, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

#
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# --- 3. TEST THREE ALPHA VALUES ---
# We check which one gives the best R2 score on the Validation set
print("--- Checking Alpha Scores ---")


lasso_01 = Lasso(alpha=0.1)
lasso_01.fit(X_train, y_train)
score_01 = r2_score(y_val, lasso_01.predict(X_val))
print(f"Alpha 0.1 Score: {round(score_01, 4)}")

lasso_001 = Lasso(alpha=0.01)
lasso_001.fit(X_train, y_train)
score_001 = r2_score(y_val, lasso_001.predict(X_val))
print(f"Alpha 0.01 Score: {round(score_001, 4)}")

lasso_0001 = Lasso(alpha=0.001)
lasso_0001.fit(X_train, y_train)
score_0001 = r2_score(y_val, lasso_0001.predict(X_val))
print(f"Alpha 0.001 Score: {round(score_0001, 4)}")


best_lasso = lasso_001
selected_features = best_lasso.coef_ != 0


X_train_final = X_train[:, selected_features]
X_val_final = X_val[:, selected_features]
X_test_final = X_test[:, selected_features]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_final, y_train)


final_predictions = model.predict(X_test_final)
final_r2 = r2_score(y_test, final_predictions)

print("\n--- Final Results ---")
print(f"Features kept: {np.sum(selected_features)}")
print(f"Final Model R2 Score: {round(final_r2, 4)}")