import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
df = pd.read_csv("/home/sumedha/Downloads/bc.csv")

print(df.describe())
print(df.keys())

x = df[
    [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean",
        "concave points_mean", "symmetry_mean", "fractal_dimension_mean",

        "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se",
        "concave points_se", "symmetry_se", "fractal_dimension_se",

        "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst", "concavity_worst",
        "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]
]

y= df["diagnosis"]

print("x_dimensions are",x.shape,"dimensions of y are:",y.shape)
#encode
y_e= y.map({"M":0,"B":1})

#convert to np.array
x=np.array(x)
y=np.array(y_e)
#train test split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)


#scale features

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

theta = np.zeros(x_train.shape[1])

# Initialize theta correctly (vector)
theta = np.zeros(x_train.shape[1])

alpha = 0.000001
iterations = 5000

# Define sigmoid once
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for i in range(iterations):
    z = np.dot(x_train, theta)
    hyp = sigmoid(z)



    #gradient of log-likelihood
    gradient = np.dot(x_train.T, (y_train - hyp))

    #update theta
    theta = theta + alpha * gradient
#prediction test
z_test = []

for i in range(len(x_test)):

    xi = x_test[i]

    zi = 0

    for j in range(len(theta)):
        zi = zi + xi[j] * theta[j]

    z_test.append(zi)

#apply sigmoid
hyp_test = []

for i in range(len(z_test)):

    zi = z_test[i]

    hi = 1 / (1 + np.exp(-zi))

    hyp_test.append(hi)

#class label
y_pred = []

for i in range(len(hyp_test)):

    if hyp_test[i] >= 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
correct = 0

for i in range(len(y_test)):

    if y_pred[i] == y_test[i]:
        correct = correct + 1

accuracy = correct / len(y_test)

print("Accuracy:", accuracy)



