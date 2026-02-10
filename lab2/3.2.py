import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



# Load data

def load_data():
    df = pd.read_csv("/home/sumedha/Downloads/simulated_dataset.csv")

    X = df[["age", "BMI", "BP", "blood_sugar"]]
    y = df["disease_score_fluct"]

    return X, y


# Split data

def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=100
    )

    print("X_train.shape:", X_train.shape)
    print("X_test.shape:", X_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape)

    return X_train, X_test, y_train, y_test


# Scale data

def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled



# Build model

def build_model(X_train_scaled, y_train):
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model



# Evaluate model

def metric_calculation(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    print("R2 score:", r2)
    return r2



# Main function

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    model = build_model(X_train_scaled, y_train)
    r2 = metric_calculation(model, X_test_scaled, y_test)


if __name__ == "__main__":
    main()










