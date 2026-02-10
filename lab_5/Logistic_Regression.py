import pandas as pd

df = pd.read_csv("/home/sumedha/Downloads/breastcancer.csv")

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

y=df["diagnosis"]

print("x_dimensions are",x.shape,"dimensions of y are:",y.shape)

#hypothesis function
