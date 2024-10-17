##########################################################
#       DBSCAN ALGORITHM => DENSITY-BASED CLUSTERING     #
##########################################################

# DBSCAN: Density-based spatial clustering of applications with noise

import numpy as np
import matplotlib.pyplot as plt

class DBSCAN:
    def __init__(self, epsilon, min_pts):
        self.epsilon = epsilon  # Maximum distance for neighborhood inclusion 
        self.min_pts = min_pts  # Minimum number of points required to form a dense region 
        self.labels = None       # Cluster labels for each point
        self.cluster_id = 0      # Current cluster ID

    def fit(self, X):
        n_samples = X.shape[0]
        self.labels = -1 * np.ones(n_samples)  # Initialize labels as -1 (noise)
        visited = np.zeros(n_samples, dtype=bool)  # Track visited points 
        
        for i in range(n_samples):
            if not visited[i]:
                visited[i] = True
                neighbors = self._get_neighbors(X, i)

                if len(neighbors) < self.min_pts:
                    self.labels[i] = -1  # Mark as noise 
                else:
                    self.cluster_id += 1  # New cluster 
                    self.labels[i] = self.cluster_id 
                    self._expand_cluster(X, neighbors, visited)

    def _get_neighbors(self, X, point_index):
        distances = np.linalg.norm(X - X[point_index], axis=1)
        return np.where(distances <= self.epsilon)[0]

    def _expand_cluster(self, X, neighbors, visited):
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True 
                new_neighbors = self._get_neighbors(X, neighbor)

                if len(new_neighbors) >= self.min_pts:
                    neighbors = np.append(neighbors, new_neighbors)

            if self.labels[neighbor] == -1:
                self.labels[neighbor] = self.cluster_id
    
    def plot_clusters(self, X):
        plt.figure(figsize=(8, 6))
        unique_labels = set(self.labels)
        for label in unique_labels:
            if label == -1:
                # Noise points
                color = 'k'  # Black for noise
                marker = 'x'
            else:
                color = plt.cm.jet(float(label) / len(unique_labels))  # Different color for each cluster 
                marker = 'o'
            
            plt.scatter(X[self.labels == label, 0], X[self.labels == label, 1], 
                        color=color, label=f'Cluster {label}' if label != -1 else 'Noise', marker=marker, edgecolor='k')

        plt.title('DBSCAN Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Sample data (2D points)
    data = np.array([
        [1, 2], [1, 4], [1, 0], [1, 5], 
        [4, 2], [4, 4], [4, 0], [4, 6],
        [8, 10], [10, 10], [10, 12], [12, 10], 
        [0, 0], [20, 0], [30, 6]
    ])

    # Create DBSCAN instance
    dbscan = DBSCAN(epsilon=5, min_pts=2)

    # Fit the model 
    dbscan.fit(data)

    # Output the cluster labels
    print("Cluster labels:", dbscan.labels)

    # Plot the clusters 
    dbscan.plot_clusters(data)
    