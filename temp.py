import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_input_hidden = np.random.randn(input_size, hidden_size)
    weights_hidden_output = np.random.randn(hidden_size, output_size)
    return weights_input_hidden, weights_hidden_output

def forward_propagation(X, weights_input_hidden, weights_hidden_output):
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = softmax(output_layer_input)

    return hidden_layer_output, output_layer_output

def backward_propagation(X, y, hidden_layer_output, output_layer_output, weights_hidden_output):
    output_error = output_layer_output - y
    output_delta = output_error

    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    return hidden_delta, output_delta

def update_weights(X, hidden_layer_output, output_delta, hidden_delta, weights_input_hidden, weights_hidden_output, learning_rate, clip_value=1.0):
    grad_output_weights = hidden_layer_output.T.dot(output_delta)
    grad_input_weights = X.T.dot(hidden_delta)

    # Gradient Clipping
    grad_output_weights = np.clip(grad_output_weights, -clip_value, clip_value)
    grad_input_weights = np.clip(grad_input_weights, -clip_value, clip_value)

    weights_hidden_output -= grad_output_weights * learning_rate
    weights_input_hidden -= grad_input_weights * learning_rate

    return weights_input_hidden, weights_hidden_output


def normalize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Check for zero standard deviation
    std[std == 0] = 1.0

    # Check for NaN values in the input data
    if np.isnan(X).any():
        raise ValueError("Input data contains NaN values.")

    normalized_X = (X - mean) / std

    return normalized_X


def encode_labels(y):
    num_classes = len(np.unique(y))
    encoded_labels = np.eye(num_classes)[y]
    return encoded_labels

def train_neural_network(X, y, hidden_size, output_size, epochs, learning_rate):
    input_size = X.shape[1]
    weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        # Forward Propagation
        hidden_layer_output, output_layer_output = forward_propagation(X, weights_input_hidden, weights_hidden_output)

        # Backward Propagation
        hidden_delta, output_delta = backward_propagation(X, y, hidden_layer_output, output_layer_output, weights_hidden_output)

        # Update Weights
        weights_input_hidden, weights_hidden_output = update_weights(X, hidden_layer_output, output_delta, hidden_delta, weights_input_hidden, weights_hidden_output, learning_rate)

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            loss = cross_entropy_loss(y, output_layer_output)
            print(f"Epoch {epoch}, Loss: {loss}")

    return weights_input_hidden, weights_hidden_output
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + epsilon)) / m
    return loss

# Example usage:
# Assuming 'X' is your input data and 'label' is your multilabel output data (between 0 and 9)
# Adjust 'hidden_size', 'output_size', 'epochs', and 'learning_rate' based on your requirements

# Generate some example data
X = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])

label = np.array([0, 1, 2, 3, 4])  # Labels between 0 and 9

# Normalize the input data
X_normalized = normalize_data(X)

# One-hot encode the labels
y_one_hot = encode_labels(label)

# Train the neural network
hidden_size = 8  # Increased hidden size
output_size = y_one_hot.shape[1]
epochs = 3000  # Increased number of epochs
learning_rate = 0.0001  # Adjusted learning rate

trained_weights_input_hidden, trained_weights_hidden_output = train_neural_network(X_normalized, y_one_hot, hidden_size, output_size, epochs, learning_rate)

print("NaN values in X:", np.isnan(X).any())
print("Infinite values in X:", np.isinf(X).any())
print("NaN values in y_one_hot:", np.isnan(y_one_hot).any())
print("Infinite values in y_one_hot:", np.isinf(y_one_hot).any())


# Perform predictions
_, predictions = forward_propagation(X_normalized, trained_weights_input_hidden, trained_weights_hidden_output)
predicted_labels = np.argmax(predictions, axis=1)
print("True Labels:", label)
print("Predicted Labels:", predicted_labels)
