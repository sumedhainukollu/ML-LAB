import numpy as np
import pandas as pd
import plotly.express as px

# 1. Create dummy data
data = {
    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
}
df = pd.DataFrame(data)


def kmeans_scratch(data, k, max_iterations=100):
    # Randomly initialize centroids by picking k points from the data
    centroids = data.sample(k).values

    for i in range(max_iterations):
        # 1. Compute distances from every point to every centroid
        # Using broadcasting for efficiency
        distances = np.linalg.norm(data.values[:, np.newaxis] - centroids, axis=2)

        # 2. Assign each point to the closest centroid
        labels = np.argmin(distances, axis=1)

        # 3. Calculate new centroids by taking the mean of assigned points
        new_centroids = np.array([data.values[labels == j].mean(axis=0) for j in range(k)])

        # Check for convergence (if centroids don't move, we are done)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return labels, centroids


# Run the function
k_value = 3
labels, final_centroids = kmeans_scratch(df, k=k_value)
df['Cluster'] = labels.astype(str)  # Convert to string for discrete color mapping

# 4. Visualize
fig = px.scatter(df, x='x', y='y', color='Cluster',
                 title=f"K-Means from Scratch (K={k_value})")

# Add final centroids to the plot
fig.add_scatter(x=final_centroids[:, 0], y=final_centroids[:, 1],
                mode='markers', marker=dict(color='black', size=15, symbol='x'),
                name="Final Centroids")

fig.show()