import numpy as np

class TwoLayerNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float = 0.01):
        """
        Initializes a basic two-layer neural network.

        :param input_size: The number of input features.
        :param hidden_size: The number of neurons in the hidden layer.
        :param output_size: The number of output neurons (usually 1 for regression or N for classification).
        :param learning_rate: Learning rate for gradient descent.
        """
        # Initialize weights and biases for the hidden and output layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Weight matrices for the hidden and output layers
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01  # Weight for input to hidden layer
        self.b1 = np.zeros((1, self.hidden_size))  # Bias for hidden layer
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01  # Weight for hidden to output layer
        self.b2 = np.zeros((1, self.output_size))  # Bias for output layer

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        :param x: Input numpy array.
        :return: Sigmoid activated output.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of the sigmoid function.

        :param x: Input numpy array.
        :return: Derivative of sigmoid.
        """
        return x * (1 - x)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        :param X: Input data (numpy array).
        :return: The output after passing through the network.
        """
        self.Z1 = np.dot(X, self.W1) + self.b1  # Input to hidden layer
        self.A1 = self.sigmoid(self.Z1)  # Activation for hidden layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Hidden to output layer
        self.A2 = self.sigmoid(self.Z2)  # Output layer activation
        return self.A2

    def backward(self, X: np.ndarray, Y: np.ndarray):
        """
        Backward pass (Gradient Descent) to update weights and biases.

        :param X: Input data (numpy array).
        :param Y: True output (numpy array).
        """
        # Compute the loss derivative with respect to output layer
        m = X.shape[0]  # Number of samples
        dA2 = self.A2 - Y  # Derivative of the loss with respect to the output
        dZ2 = dA2 * self.sigmoid_derivative(self.A2)  # Derivative of loss w.r.t Z2

        # Gradients for W2 and b2
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Backpropagate through the hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)  # Derivative of loss w.r.t Z1

        # Gradients for W1 and b1
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases using gradient descent
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 1000):
        """
        Train the neural network using gradient descent.

        :param X: Training data (numpy array).
        :param Y: True labels (numpy array).
        :param epochs: Number of training iterations.
        """
        for epoch in range(epochs):
            self.forward(X)  # Forward pass
            self.backward(X, Y)  # Backward pass
            if epoch % 100 == 0:
                loss = np.mean(np.square(Y - self.A2))  # Mean Squared Error (MSE) loss
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        :param X: Input data.
        :return: Predicted output.
        """
        return self.forward(X)

# Example usage
if __name__ == "__main__":
    # XOR problem example for a binary classification
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
    Y = np.array([[0], [1], [1], [0]])  # Expected outputs (XOR)

    # Initialize the neural network with 2 input neurons, 2 hidden neurons, and 1 output neuron
    nn = TwoLayerNN(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)
    
    # Train the network
    nn.train(X, Y, epochs=1000)

    # Test the network
    predictions = nn.predict(X)
    print("Predictions:", predictions)
