from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

# Load data
iris = load_iris()
x = iris.data
print(x.shape)

y = iris.target
print(y.shape)

# Model training
model = LinearRegression()

# K-fold validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

scores = cross_val_score(model, x, y, cv=kf)

print("R2 scores for each fold:", scores)
print("Average R2 score:", scores.mean())
