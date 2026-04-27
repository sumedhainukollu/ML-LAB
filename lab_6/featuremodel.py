import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score



data = fetch_california_housing()
X = data.data
y = data.target


X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

selected = lasso.coef_ != 0

X_train = X_train[:, selected]
X_val = X_val[:, selected]
X_test = X_test[:, selected]


models = {
    "Linear": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42)
}

best_model = None
best_score = -np.inf

for name, model in models.items():
    model.fit(X_train, y_train)
    score = r2_score(y_val, model.predict(X_val))
    print(name, "Validation R²:", round(score, 4))

    if score > best_score:
        best_score = score
        best_model = model
        best_name = name

print("\nBest Model:", best_name)

test_score = r2_score(y_test, best_model.predict(X_test))
print("Test R²:", round(test_score, 4))