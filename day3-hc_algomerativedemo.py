import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate toy dataset
X, y_true = make_blobs(n_samples=50, centers=3, cluster_std=0.6, random_state=42)

# Agglomerative clustering (just to compare later)
agg = AgglomerativeClustering(n_clusters=3)
labels = agg.fit_predict(X)

# ---- Plot clustering results ----
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=40)
plt.title("Agglomerative Clustering (3 clusters)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# ---- Plot dendrogram ----
# Use 'ward' linkage for hierarchical tree
Z = linkage(X, method="ward")

plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90., leaf_font_size=10., show_contracted=True)
plt.title("Dendrogram (Hierarchical Clustering)")
plt.xlabel("Cluster Size")
plt.ylabel("Distance")
plt.show()
