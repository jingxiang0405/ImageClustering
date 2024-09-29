from sklearn.cluster import KMeans


def apply_kmeans(features, nclusters=5):
    kmeans = KMeans(n_clusters=nclusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels
