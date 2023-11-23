import numpy as np
# One hot encoding of labels
def encode_labels(y):
    # One-hot encoding for labels between 0 and 9
    num_classes = len(np.unique(y))
    encoded_labels = np.eye(num_classes)[y]
    return encoded_labels
