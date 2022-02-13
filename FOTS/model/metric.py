from difflib import SequenceMatcher

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_char_similarity(pred, gt):
    return SequenceMatcher(None, pred, gt).ratio()

def get_mean_char_similarity(pred, gt):
    return np.mean([get_char_similarity(p, g) for p, g in zip(pred, gt)])

def mae(pred, gt):
    return mean_absolute_error(gt, pred)

def rmse(pred, gt):
    return mean_squared_error(gt, pred, squared=False)