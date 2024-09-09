import numpy as np

class LinearRegression:
    def __init__(self, alpha=0.0, iterations=1000, learning_rate=0.01, lasso=False, ridge=False):
        self.alpha = alpha  # Regularization strength
        self.iterations = iterations  # Number of iterations for gradient descent
        self.learning_rate = learning_rate  # Learning rate
        self.lasso = lasso  # Flag for Lasso regularization
        self.ridge = ridge  # Flag for Ridge regularization

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)  # Initialize weights (excluding bias)
        self.bias = 0  # Initialize bias term

        for _ in range(self.iterations):
            # Make predictions
            predictions = X.dot(self.weights) + self.bias
            errors = predictions - y

            # Compute gradients
            dw = (2/m) * X.T.dot(errors)  # Derivative w.r.t weights
            db = (2/m) * np.sum(errors)  # Derivative w.r.t bias

            # Add regularization terms
            if self.lasso:
                dw += self.alpha * np.sign(self.weights)  # Lasso regularization
            elif self.ridge:
                dw += 2 * self.alpha * self.weights  # Ridge regularization

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return X.dot(self.weights) + self.bias

