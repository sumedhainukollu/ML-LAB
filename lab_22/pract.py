import numpy as np
import pandas as pd
from ISLP import load_data

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


data = load_data('NCI60')

X = pd.DataFrame(data['data'])
y = pd.DataFrame(data['labels'], columns=['label'])


le = LabelEncoder()
y_encoded = le.fit_transform(y['label']).ravel()



print( X.head())
print( y.head())


agglo = FeatureAgglomeration(n_clusters=40)
X_agg = agglo.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_agg, y_encoded, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model_hc = SVC(kernel='linear')
model_hc.fit(X_train, y_train)

pred_hc = model_hc.predict(X_test)

print("hierachaial clustering")
print("Accuracy:", accuracy_score(y_test, pred_hc))
print(classification_report(y_test, pred_hc))


pca = PCA(n_components=40)
X_pca = pca.fit_transform(X_scaled)


X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)

model_pca = SVC(kernel='linear')
model_pca.fit(X_train, y_train)

pred_pca = model_pca.predict(X_test)

print("PCA ")
print("Accuracy:", accuracy_score(y_test, pred_pca))
print(classification_report(y_test, pred_pca))
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

plt.figure(figsize=(16,8))

sch.dendrogram(
    sch.linkage(X_scaled, method='ward'),
    truncate_mode='level',
    p=5
)

plt.title("dendrogram ")
plt.xlabel("feature Clusters")
plt.ylabel("distance")

plt.show()