import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# STEP 1: Load your dataset
# Replace with your file or dataframe
df = pd.read_csv("your_dataset.csv")

# Optional: drop non-numeric columns (PCA hates text like I hate vague questions)
df_numeric = df.select_dtypes(include=[np.number])

# STEP 2: Standardize the data (VERY IMPORTANT)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# STEP 3: Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# STEP 4: Explained Variance
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

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
plt.ylabel("Cumulative Variance")
plt.title("Cumulative Variance Explained")
plt.show()

# STEP 7: PCA Scatter Plot (First 2 PCs)
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection")
plt.show()