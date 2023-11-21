import numpy as np
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