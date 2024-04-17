**Title: K-Nearest Neighbors (KNN) Classifier Implementation**

**Description:**
This Python code implements a K-Nearest Neighbors (KNN) classifier, a simple yet powerful algorithm used for classification tasks. KNN is a non-parametric and instance-based learning algorithm that memorizes the training dataset. Given a new, unseen data point, it predicts the class label by finding the majority class among its k nearest neighbors in the feature space.

**Key Components:**
1. **Initialization:** The class `KNN` is initialized with a parameter `k`, which determines the number of neighbors considered for classification. By default, `k` is set to 3.
   
2. **Model Fitting:** The `fit` method stores the training data (`X_train` and `y_train`), where `X_train` represents the features and `y_train` represents the corresponding labels.

3. **Prediction:** The `predict` method predicts the labels for new data points provided in `X`. For each data point, it calculates the distances to all training data points, identifies the `k` nearest neighbors, and predicts the label based on the majority class among these neighbors.

4. **Accuracy Calculation:** The `score` method evaluates the accuracy of the model by comparing the predicted labels with the actual labels (`y_test`). It returns the ratio of correct predictions to the total number of predictions.

5. **Helper Method:** The `_predict` method is a helper function that predicts the label for a single data point. It calculates distances to all training data points, selects the `k` nearest neighbors, and determines the most common label among them.

**Usage:**
To utilize this KNN classifier:
- Initialize the `KNN` class with an optional parameter `k`.
- Use the `fit` method to train the model with training data (`X_train` and `y_train`).
- Employ the `predict` method to predict labels for new data points (`X_test`).
- Evaluate the model's accuracy using the `score` method with test data (`X_test` and `y_test`).

This implementation provides a flexible and efficient way to perform classification tasks using the KNN algorithm, allowing users to easily integrate it into their projects or machine learning pipelines.
