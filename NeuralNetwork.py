import numpy as np


class FCLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)
        self.bias = np.random.randn(1, output_size) / np.sqrt(input_size + output_size)

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.reshape(-1,1), output_error.reshape(1,-1))
        # bias_error = output_error

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


class ActivationLayer:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, output_error, learning_rate):
        return output_error * self.activation_prime(self.input)


class FlattenLayer:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward(self, input):
        return np.reshape(input, (1, -1))

    def backward(self, output_error, learning_rate):
        return np.reshape(output_error, self.input_shape)


class SoftmaxLayer:
    def __init__(self, input_size):
        self.input_size = input_size

    def forward(self, input):
        self.input = input
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_error, learning_rate):
        input_error = np.zeros(output_error.shape)
        out = np.tile(self.output.T, self.input_size)
        return self.output * np.dot(output_error, np.identity(self.input_size) - out)







# Your data
data_str = """
00000000000011000000000000000000
00000000000111101000000000000000
00000000001111111111000000000000
00000000001111111111100000000000
00000000011111111111111000000000
00000000011111111111111110000000
00000000011111110011111111000000
00000000111111100000111111000000
00000000111110000000011111000000
00000001111110000000011111000000
00000001111100000000001111000000
00000001111000000000011111000000
00000001111000000000001111000000
00000001111000000000001111000000
00000001111000000000001111000000
00000001111000000000001111000000
00000001110000000000001111000000
00000001110000000000001111000000
00000011110000000000001111000000
00000001110000000000011111000000
00000001110000000000001111000000
00000001111000000000011111000000
00000001111000000000011111000000
00000001111000000001111110000000
00000001111000000001111100000000
00000001111100000011111100000000
00000000111110000111111000000000
00000000011111111111100000000000
00000000011111111111000000000000
00000000001111111111000000000000
00000000000111111100000000000000
00000000000000100000000000000000
 0
"""

# Split the data into individual samples
samples = data_str.strip().split("\n\n")

# Initialize empty lists for features (X) and labels (y)
X = []
y = []

# Process each sample
for sample in samples:
    lines = sample.strip().split("\n")

    # Extract label
    label = int(lines[-1])

    # Convert binary strings into numerical arrays
    binary_array = np.array([list(map(int, line.replace("0", "0 ").replace("1", "1 ").split())) for line in lines[:-1]])

    # Flatten the 2D array into a 1D array and append to X
    X.append(binary_array.flatten())

    # Append the label to y
    y.append(label)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)


# Print the shapes of X and y
print("X shape:", X.shape)
print("y shape:", y.shape)

# Optionally, save the datasets to files if needed
np.save("feature.npy", X)
np.save("label.npy", y)

X_train = np.load('feature.npy')
y_train = np.load('label.npy')

network = [
    FlattenLayer(input_shape=(1024, 1)),
    FCLayer(1024, 1024),
    ActivationLayer(relu, relu_prime),
    FCLayer(1024, 10),
    SoftmaxLayer(1)
]


epochs = 10
learning_rate = 0.1

# training
for epoch in range(epochs):
    loss = 0.0
    output_error = 0.0
    for x, y_true in zip(X_train, y_train):
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)

        # error (display purpose only)
        loss += sse(y_true, output)

        # backward
        output_error = mse_prime(y_true, output)
        for layer in reversed(network):
            output_error = layer.backward(output_error, learning_rate)

    loss /= len(X_train)
    print('%d/%d, Loss=%f' % (epoch + 1, epochs, loss))