import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torchmetrics.functional import explained_variance

df=pd.read_csv("/home/sumedha/Downloads/All_Datasets/Kaggle_datasets/breast-cancer.csv")
df_numeric = df.select_dtypes(include=[np.number])
scaler=StandardScaler()
x_scaled=scaler.fit_transform(df_numeric)
pca=PCA()
x_pca=pca.fit_transform(df_numeric)
print("explained variance :",pca.explained_variance_ratio_)
print("cumulative variance", np.cumsum(pca.explained_variance_ratio_))

#plot
plt.figure()
plt.scatter(x_pca[:,0],x_pca[:,1])
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.title("pca projection")


# STEP 5: Scree Plot
plt.figure()
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_,
         marker='o')
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.title("Scree Plot")
plt.show()

# STEP 6: Cumulative Variance
plt.figure()
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         np.cumsum(pca.explained_variance_ratio_),
         marker='o')
plt.xlabel("Principal Component")
plt.ylabel("cumulative Variance Explained")
plt.title("Scree Plot")
plt.show()


