import numpy as np


def binary_to_one_hot(binary_sequence, label):
    # Convert the binary sequence to a one-hot encoding
    one_hot_encoding = [0] * 10
    one_hot_encoding[label] = 1.

    return one_hot_encoding


# Read sequences and labels from file
with open('file.txt', 'r') as file:
    lines = file.readlines()

# Assuming labels are provided as [0, 1, 2, ..., 9, 0, 1, ...]
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * (len(lines) // 10)

# Convert sequences to one-hot encoding
one_hot_data = [binary_to_one_hot(line.strip(), label) for line, label in zip(lines, labels)]

# Split data into training and testing sets
split_ratio = 0.8  # 80% training, 20% testing
split_index = int(len(one_hot_data) * split_ratio)

train_data = one_hot_data[:split_index]
test_data = one_hot_data[split_index:]

train_labels = labels[:split_index]
test_labels = labels[split_index:]


# Convert lists to NumPy arrays
train_data_np = np.array(train_data)
test_data_np = np.array(test_data)

# Save NumPy arrays to .npy files
np.save('train/X_train.npy', train_data_np)
np.save('test/X_test.npy', test_data_np)

# Save labels as well
np.save('train/y_train.npy', np.array(train_labels))
np.save('test/y_test.npy', np.array(test_labels))


