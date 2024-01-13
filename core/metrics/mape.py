import numpy as np

def mape(y_pred, y_true): # mape = mean absolute percentage error
  return (np.abs(y_pred - y_true)/(y_true+np.e)).mean() * 100