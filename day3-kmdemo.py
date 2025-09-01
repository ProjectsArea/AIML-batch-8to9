# sci-kit-learn: k-means example
# ml library - 

import numpy as np # numerical operations - matrix operations
import matplotlib.pyplot as plt # plotting - 2D plots

from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs # to generate toy datasets

# Generate a toy dataset with 3 clusters
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)

# # Fit KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X) # fit the model - common to all ml algorithms

# # Get cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Plot the results
plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], X[:, 1], s=30, cmap='viridis')
# plt.show()
plt.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.7, marker='X')
plt.title("KMeans Clustering Example")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
