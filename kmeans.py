import numpy as np

class KMeansCustom:
    def __init__(self, K=3, distance_metric='euclidean', init='random', max_iters=300, tol=1e-4, random_state=None):
        self.K = K
        self.distance_metric = distance_metric
        self.init = init
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
           
    def _init_kmeans_plus_plus(self, X):
        # K-Means++ Initialization
        n_samples, n_features = X.shape
        centroids = np.empty((self.K, n_features))
        # Choose the first centroid randomly
        centroids[0] = X[np.random.randint(n_samples)]
        
        # Compute the remaining centroids
        for k in range(1, self.K):
            distances = np.min(np.linalg.norm(X[:, np.newaxis] - centroids[:k], axis=2), axis=1)
            probabilities = distances / distances.sum()
            next_centroid = X[np.random.choice(n_samples, p=probabilities)]
            centroids[k] = next_centroid
        
        return centroids

    def fit(self, X):
        np.random.seed(self.random_state)
        
        # K-Means++ Initialization
        if self.init == 'kmeans++':
            centroids = self._init_kmeans_plus_plus(X)
        else:
            # Random initialization
            centroids = X[np.random.choice(X.shape[0], self.K, replace=False)]
        
        labels = np.zeros(X.shape[0])
        for _ in range(self.max_iters):
            # Compute distances
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            # Assign labels based on closest centroid
            new_labels = np.argmin(distances, axis=1)

            # Recompute centroids
            new_centroids = np.array([X[new_labels == k].mean(axis=0) for k in range(self.K)])
            
            # Check for convergence
            if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < self.tol):
                break

            labels = new_labels
            centroids = new_centroids

        return centroids, labels
