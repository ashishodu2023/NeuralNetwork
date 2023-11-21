import numpy as np
from Activations import sigmoid, sigmoid_derivative,softmax
def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_input_hidden = np.random.randn(input_size, hidden_size)
    weights_hidden_output = np.random.randn(hidden_size, output_size)
    return weights_input_hidden, weights_hidden_output

def forward(X, weights_input_hidden, weights_hidden_output):
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = softmax(output_layer_input)

    return hidden_layer_output, output_layer_output

def backward(X, y, hidden_layer_output, output_layer_output, weights_hidden_output):
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