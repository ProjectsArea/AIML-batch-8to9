import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Generate a toy dataset (moons shape, good for DBSCAN)
X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)

# Fit DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)  # eps = neighborhood size, min_samples = density threshold
labels = dbscan.fit_predict(X)

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30)
# plt.scatter(X[:, 0], X[:, 1], cmap="viridis", s=30)

plt.title("DBSCAN Clustering Example")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
