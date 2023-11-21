import numpy as np
def encode_labels(y):
    # One-hot encoding for labels between 0 and 9
    num_classes = len(np.unique(y))
    encoded_labels = np.eye(num_classes)[y]
    return encoded_labels
