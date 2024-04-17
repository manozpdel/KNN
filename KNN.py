# Importing necessary libraries
from collections import Counter
import numpy as np

# Defining a class named KNN
class KNN:
    # Constructor to initialize the class with a parameter k (number of neighbors)
    def __init__(self, k=3):
        self.k = k

    # Method to fit the model, storing training data
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Method to make predictions on new data
    def predict(self, X):
        # For each data point in X, predict the label
        predictions = [self._predict(x) for x in X.values]
        return np.array(predictions)

    # Method to calculate the accuracy of the model
    def score(self, X_test, y_test):
        # Make predictions on the test data
        predictions = self.predict(X_test)
        # Calculate accuracy by comparing predicted labels to actual labels
        accuracy = np.sum(predictions == y_test) / len(y_test)
        return accuracy

    # Helper method to predict the label of a single data point
    def _predict(self, x):
        # Calculate distances from the input data point to all training data points
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train.values]
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Get labels of k nearest neighbors
        k_nearest_labels = [self.y_train.iloc[i] for i in k_indices]
        # Find the most common label among the k nearest neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
