from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
data = pd.read_csv("/home/sumedha/Downloads/All_Datasets/Kaggle_datasets/diabetes (1).csv")
data = data.dropna()

print(data.head())

# Encode target
le = LabelEncoder()
data["Outcome"] = le.fit_transform(data["Outcome"])

# Features & target
x = data.drop("Outcome", axis=1)
y = data["Outcome"]

# ---------------- EDA ---------------- #
def eda(x, y):
    print("Shape:", data.shape)

    corr = data.corr()
    print(corr)

    print("\nMissing values:\n", data.isnull().sum())

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

    plt.figure(figsize=(12,6))
    sns.boxplot(data=data)
    plt.xticks(rotation=45)
    plt.title("Boxplot for Outlier Detection")
    plt.show()

    data.hist(figsize=(12,10))
    plt.show()

    return x, y


# ---------------- Scaling ---------------- #
def scale_data(x):
    sc = StandardScaler()
    x_scaled = sc.fit_transform(x)
    return x_scaled


# ---------------- Model ---------------- #
def build_model():
    base_model = DecisionTreeRegressor(max_depth=5, min_samples_split=5)

    model = BaggingRegressor(
        estimator=base_model,
        n_estimators=50,
        random_state=42
    )
    return model


# ---------------- Cross Validation ---------------- #
def cross_validation(model, x_scaled, y):
    scores_r2 = cross_val_score(model, x_scaled, y, cv=5, scoring='r2')
    scores_mse = cross_val_score(model, x_scaled, y, cv=5, scoring='neg_mean_squared_error')

    print("\n--- Cross Validation ---")
    print("Mean R2:", scores_r2.mean())
    print("Mean MSE:", -scores_mse.mean())


# ---------------- Hyperparameter Tuning ---------------- #
def hyperparameter_tuning(x_scaled, y):

    model = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        random_state=42
    )

    param_grid = {
        'n_estimators': [10, 50, 100],
        'estimator__max_depth': [3, 5, 10],
        'estimator__min_samples_split': [2, 5, 10]
    }

    grid = GridSearchCV(model, param_grid, cv=5, scoring='r2')
    grid.fit(x_scaled, y)

    print("\n--- Best Parameters ---")
    print(grid.best_params_)

    return grid.best_estimator_


# ---------------- Main ---------------- #
def main():
    eda(x, y)

    x_scaled = scale_data(x)

    model = build_model()

    cross_validation(model, x_scaled, y)

    best_model = hyperparameter_tuning(x_scaled, y)

    # Evaluate best model using cross-val again
    print("\n--- After Tuning (Cross-Val) ---")
    cross_validation(best_model, x_scaled, y)


main()