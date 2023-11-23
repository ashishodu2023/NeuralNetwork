# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 02:13:28 2023

@author: Ashish
"""

import numpy as np
def encode_labels(y):
    # One-hot encoding for labels between 0 and 9
    num_classes = len(np.unique(y))
    encoded_labels = np.eye(num_classes)[y]
    return encoded_labels
