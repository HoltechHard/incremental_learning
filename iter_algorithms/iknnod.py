import numpy as np
from collections import deque
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

class IncrementalKNNOutlierDetector:
    def __init__(self, k=3, threshold_percentile=75):
        self.k = k 
        self.threshold_percentile = threshold_percentile 
        self.data = []  # List to store data points
        self.tree = None  # KD-Tree for efficient neighbor queries
        self.outlier_scores = []  # List to store outlier scores 

    def add_data_point(self, new_point):
        self.data.append(new_point)
        self.update_tree()
        self.update_outlier_scores(new_point)

    def update_tree(self):
        if len(self.data) > 0:
            self.tree = KDTree(self.data)

    def update_outlier_scores(self, new_point):
        if len(self.data) < self.k:
            return  # Not enough points to calculate outlier scores

        # Find the k-nearest neighbors
        distances, indices = self.tree.query([new_point], k=self.k)
        distances = distances[0]

        # Calculate the outlier score based on distances
        score = np.mean(distances)  # Example score calculation 

        # Update the threshold dynamically
        if len(self.outlier_scores) >= self.k:
            threshold = np.percentile(self.outlier_scores, self.threshold_percentile)
            if score > threshold:
                print(f"Point {new_point} is an outlier with score {score}")
    
    def get_outliers(self):
         # Return current outliers based on updated scores
        if len(self.outlier_scores) < self.k:
            return [], None  # Not enough scores to calculate threshold
        
        # Return current outliers based on updated scores
        threshold = np.percentile(self.outlier_scores, self.threshold_percentile)
        return [index for index, score in enumerate(self.outlier_scores) if score > threshold], threshold
    
    def plot_outliers(self):
        # Convert data to a numpy array for easy indexing 
        X = np.array(self.data)
        # Get outlier indices and threshold
        outlier_indices, threshold = self.get_outliers()
        
        # Create a mask for outliers
        outliers = np.array(outlier_indices)
        
        # Plot the data points and highlight outliers
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], color='blue', label='Normal Points')
        plt.scatter(X[outliers, 0], X[outliers, 1], color='red', label='Outliers', marker='x', s=100)

        plt.title('K-Nearest Neighbors Outlier Detection')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.axhline(y=threshold, color='green', linestyle='--', label='Threshold')
        plt.legend()
        plt.grid()
        plt.show()        


# Example usage
if __name__ == "__main__":
    detector = IncrementalKNNOutlierDetector(k=3)
    data_points = np.array([
        [1, 2], [1, 4], [1, 0], [1, 5], 
        [4, 2], [4, 4], [4, 0], [4, 6],
        [8, 10], [10, 10], [10, 12], [12, 10], 
        [0, 0], [20, 0], [1000, 6]
    ])
    
    for point in data_points:
        detector.add_data_point(point)

    outliers = detector.get_outliers()
    print("Detected outliers:", outliers)

    detector.plot_outliers()
