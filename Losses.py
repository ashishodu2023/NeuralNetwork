import numpy as np

# Mean absolute error loss
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Cross Entropy loss
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + epsilon)) / m
    return loss
