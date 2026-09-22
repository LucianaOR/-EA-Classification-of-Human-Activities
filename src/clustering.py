import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

# -------------------------------
# Função: K-Means Outliers
# -------------------------------
def kmeans_outliers(df, variables=["acc_x", "acc_y", "acc_z"], n_clusters=3, title="K-Means Outliers",
                    outlier_method='zscore', z_thresh=3.0, percentile=95):

    data = df[variables].dropna().values

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    dists = np.linalg.norm(data - centers[labels], axis=1)

    outliers = np.zeros_like(dists, dtype=bool)
    if outlier_method == 'percentile':
        threshold = np.percentile(dists, percentile)
        outliers = dists > threshold
    elif outlier_method == 'zscore':
        mean = np.mean(dists)
        std = np.std(dists)
        if std == 0:
            outliers = np.zeros_like(dists, dtype=bool)
            threshold = None
        else:
            z = (dists - mean) / std
            outliers = np.abs(z) > z_thresh
            threshold = z_thresh
    else:
        raise ValueError("outlier_method deve ser 'zscore' ou 'percentile'.")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[~outliers, 0], data[~outliers, 1], data[~outliers, 2],
               c=labels[~outliers], cmap='tab10', s=10, label="Inliers")

    ax.scatter(data[outliers, 0], data[outliers, 1], data[outliers, 2],
               c='red', s=20, label="Outliers")

    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
               c='black', marker='X', s=100, label='Centroides')

    ax.set_xlabel(variables[0])
    ax.set_ylabel(variables[1])
    ax.set_zlabel(variables[2])
    ax.set_title(f"{title} ({n_clusters} clusters)")
    ax.legend()
    plt.show()

    pct = (np.sum(outliers) / len(outliers) * 100) if len(outliers) else 0
    print(f"Detetados {np.sum(outliers)} outliers de {len(outliers)} pontos (~{pct:.2f}%)")

    return {
        "labels": labels,
        "centers": centers,
        "outliers": outliers,
        "distances": dists,
        "threshold_used": threshold,
        "method": outlier_method
    }

# -------------------------------
# Função: DBSCAN Outliers
# -------------------------------
def dbscan_outliers(df, variables=["acc_x", "acc_y", "acc_z"], eps=0.5, min_samples=10):
    data = df[variables].dropna().values
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"DBSCAN: {n_clusters} clusters encontrados, {n_noise} outliers.")

    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "n_noise": n_noise
    }


# -------------------------------
# Função: Plot DBSCAN Results
# -------------------------------
def plot_dbscan_results(data, labels, variables=["x", "y", "z"], title="DBSCAN Clustering"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = set(labels)
    for lbl in unique_labels:
        mask = labels == lbl
        if lbl == -1:
            ax.scatter(data[mask, 0], data[mask, 1], data[mask, 2], c='red', s=20, label='Outliers')
        else:
            ax.scatter(data[mask, 0], data[mask, 1], data[mask, 2], s=10, label=f'Cluster {lbl}')

    ax.set_xlabel(variables[0])
    ax.set_ylabel(variables[1])
    ax.set_zlabel(variables[2])
    ax.set_title(title)
    ax.legend()
    plt.show()
