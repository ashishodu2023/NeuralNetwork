import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_pred.size

def mae(y_true, y_pred):
    errors = np.abs(y_true - y_pred)
    mae = np.mean(errors)
    return mae

def mae_prime(y_true, y_pred):
    return np.sign(y_pred - y_true)
def sse(y_true, y_pred):
    return 0.5 * np.sum(np.power(y_true - y_pred, 2))

def sse_prime(y_true, y_pred):
    return y_pred - y_true


def sparse_categorical_crossentropy(y_true, y_pred):
    # convert true labels to one-hot encoding
    y_true_onehot = np.zeros_like(y_pred)
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    loss = -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=-1))
    return loss


def sparse_categorical_crossentropy_prime(y_true, y_pred):
    # Convert true labels to one-hot encoding
    y_true_onehot = np.zeros_like(y_pred)
    y_true_onehot[np.arange(len(y_true)), y_true] = 1

    # Calculate derivative
    derivative = -(y_true_onehot - y_pred) / len(y_true)

    return derivative
