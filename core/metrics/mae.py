import numpy as np


def mae(y_pred, y_true):
  return np.abs(y_pred - y_true).mean()
