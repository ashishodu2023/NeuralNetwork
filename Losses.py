# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 02:12:54 2023

@author: Ashish
"""

import numpy as np

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + epsilon)) / m
    return loss
