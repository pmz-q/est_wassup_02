import numpy as np

def mape(y_pred, y_true): # mape = mean absolute percentage error
  return (np.abs(y_pred - y_true)/y_true).mean() * 100

def mae(y_pred, y_true):
  return np.abs(y_pred - y_true).mean()

def r2_score(y_pred, y_true):
    mean_y_true = np.mean(y_true)
    ss_total = np.sum((y_true - mean_y_true)**2) # SST
    ss_residual = np.sum((y_true - y_pred)**2) # SSR
    r2 = 1 - (ss_residual / ss_total)
    return r2
# R2 추가 
# MSE 추가