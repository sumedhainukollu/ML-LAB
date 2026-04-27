import numpy as np
import plotly.express as px
import pandas as pd
data= {
    'x': [11,20,32,45,66,77,88],
    'y':[22,33,44,55,34,100,200]
}
df=pd.DataFrame(data)


def k_means(data,k,max_iterations=100):
    centroids=data.sample(k).values
    for i in range(max_iterations):
        distances=np.linalg.norm(data.values[:,np.newaxis]-centroids, axis=2)
        labels=np.argmin(distances,axis=1)
        for j in range (k):
            new_centroids = np.array([data.values[labels == j].mean(axis=0))

            if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

