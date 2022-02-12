from difflib import SequenceMatcher

import numpy as np
from sklearn.metrics import mean_squared_error

from ..utils.eval_tools.icdar2015 import eval as icdar_eval

def fots_metric(pred, gt):
    config = icdar_eval.default_evaluation_params()
    output = icdar_eval.eval(pred, gt, config)
    return output['method']['precision'], output['method']['recall'], output['method']['hmean']

def rmse(pred, gt):
    return mean_squared_error(gt, pred, squared=False)

def get_char_similarity(pred, gt):
    return SequenceMatcher(None, pred, gt).ratio()

def get_mean_char_similarity(pred, gt):
    return np.mean([get_char_similarity(p, g) for p, g in zip(pred, gt)])