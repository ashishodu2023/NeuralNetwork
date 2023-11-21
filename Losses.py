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


def cross_entropy_loss(y_true, y_pred):
    E = 1e-15  # Small constant to avoid numerical instability (log(0))
    y_pred = np.clip(y_pred, E, 1 - E)  # Clip to avoid log(0)
    # Ensure y_true is a numpy array
    y_true = np.array(y_true)
    # Calculate cross-entropy loss
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss

def cross_entropy_loss_prime(y_true, y_pred):
    E = 1e-5  # Small constant to avoid numerical instability (log(0))
    y_pred = np.clip(y_pred, E, 1 - E)  # Clip to avoid log(0)
    loss = -y_true / y_pred
    return loss
