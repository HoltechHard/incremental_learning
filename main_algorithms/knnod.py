#################################################
#       KNNOD ALGORITHM => OUTLIER DETECTION    #
#################################################

# KNNOD: K-Nearest Neighbors outlier detection

import numpy as np
import matplotlib.pyplot as plt

class KNNOutlierDetector:
    def __init__(self, k=5, threshold=2):
        self.k = k  # Number of nearest neighbors
        self.threshold = threshold  # Threshold for outlier detection 
        self.outliers_ = None  # List to store outlier points 
        
    def fit(self, X):
        # Fit the model and identify outliers
        self.X = X
        self.n_samples = X.shape[0]

        # Calculate distances and find k-nearest neighbors
        distances = np.zeros((self.n_samples, self.n_samples))

        for i in range(self.n_samples):
            for j in range(self.n_samples):
                if i != j:
                    distances[i, j] = self.euclidean_distance(X[i], X[j])

        # Calculate average distance to k-nearest neighbors
        avg_distances = np.zeros(self.n_samples)

        for i in range(self.n_samples):
            # Get the indices of the k-nearest neighbors
            k_nearest_indices = np.argsort(distances[i])[:self.k + 1][1:]  # Exclude the point itself
            avg_distances[i] = np.mean(distances[i, k_nearest_indices])

        # Calculate the mean distance across all points
        mean_distance = np.mean(avg_distances)

        # Calculate outlier scores 
        self.outlier_scores = avg_distances / mean_distance

        # Identify outliers based on the threshold
        self.outliers_ = np.where(self.outlier_scores > self.threshold)[0]

    def euclidean_distance(self, point1, point2):
        # Calculate the Euclidean distance between two points
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def get_outliers(self):
        # Return the indices of outliers
        return self.outliers_

    def get_outlier_scores(self):
        # Return the outlier scores
        return self.outlier_scores

    def plot_outliers(self):
        # Plot the data points and highlight outliers
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X[:, 0], self.X[:, 1], color='blue', label='Normal Points')
        plt.scatter(self.X[self.outliers_, 0], self.X[self.outliers_, 1], color='red', 
                    label='Outliers', marker='x', s=100)

        plt.title('K-Nearest Neighbors Outlier Detection')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.grid()
        plt.show()

# Example usage
if __name__ == "__main__":
    # dataset
    data = np.array([
        [1, 2], [1, 4], [1, 0], [1, 5], 
        [4, 2], [4, 4], [4, 0], [4, 6],
        [8, 10], [10, 10], [10, 12], [12, 10], 
        [0, 0], [20, 0], [30, 6]
    ])

    # Create KNN Outlier Detector instance 
    knn_outlier_detector = KNNOutlierDetector(k=3, threshold=2)

    # Fit the model 
    knn_outlier_detector.fit(data)

    # Get outliers
    outliers = knn_outlier_detector.get_outliers()
    outlier_scores = knn_outlier_detector.get_outlier_scores()

    # Output the results
    print("Indices of outliers:", outliers)
    print("Outlier scores:", outlier_scores)

    # Plot the outliers
    knn_outlier_detector.plot_outliers()
