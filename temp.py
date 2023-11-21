import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def sigmoid_derivative(x):
    return x * (1 - x)

def cross_entropy_loss(y, y_pred):
    return -np.sum(y * np.log(y_pred + 1e-15)) / len(y)

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    bias_hidden = np.zeros((1, hidden_size))
    bias_output = np.zeros((1, output_size))
    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

def forward_propagation(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = softmax(output_layer_input)

    return hidden_layer_output, output_layer_output

def backward_propagation(X, y, hidden_layer_output, output_layer_output,
                         weights_hidden_output, bias_hidden, bias_output, learning_rate):
    weights_input_hidden = 0
    output_error = output_layer_output - y
    hidden_error = output_error.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

    weights_hidden_output -= hidden_layer_output.T.dot(output_error) * learning_rate
    bias_output -= np.sum(output_error, axis=0, keepdims=True) * learning_rate

    weights_input_hidden -= X.T.dot(hidden_error) * learning_rate
    bias_hidden -= np.sum(hidden_error, axis=0, keepdims=True) * learning_rate

def train(X, y, hidden_size, output_size, epochs, learning_rate, batch_size):
    input_size = X.shape[1]
    weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = initialize_parameters(
        input_size, hidden_size, output_size)

    num_batches = len(X) // batch_size

    for epoch in range(epochs):
        for batch in range(num_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]

            hidden_layer_output, output_layer_output = forward_propagation(
                X_batch, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

            backward_propagation(X_batch, y_batch, hidden_layer_output, output_layer_output,
                                weights_hidden_output, bias_hidden, bias_output, learning_rate)

        if epoch % 100 == 0:
            loss = cross_entropy_loss(y, output_layer_output)
            print(f"Epoch {epoch}, Loss: {loss}")

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

def predict(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    _, output = forward_propagation(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
    return np.argmax(output, axis=1)

# Load the data
X_train = np.load('train/X_train.npy')
y_train = np.load('train/y_train.npy')

# Convert labels to one-hot encoding
num_classes = 10
y_train_one_hot = np.eye(num_classes)[y_train]

# Set random seed for reproducibility
np.random.seed(42)

# Set hyperparameters
input_size = X_train.shape[1]
hidden_size = 32
output_size = 10
learning_rate = 0.01
epochs = 100
batch_size = 32

# Initialize weights
weights_input_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)

weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = train(
    X_train, y_train_one_hot, hidden_size, output_size, epochs, learning_rate, batch_size)

# Make predictions
X_test = np.random.rand(10, X_train.shape[1])
predictions = predict(X_test, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

print("Predictions:", predictions)
