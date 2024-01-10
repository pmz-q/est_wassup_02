import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def nn_predict(
  tst_dl: DataLoader, model: nn.Module,
  test_size: int, predicton_size: int, window_size:int,
  device
):
  result = np.array([])
  len_result = 0
  for x, _ in tst_dl:
    # add predictions as x['target']
    if len_result > 0:
      if len_result <= window_size:
        x[:,-len_result:,0] = torch.Tensor(result) # batch, window, n feature
      else:
        x[:,:,0] = torch.Tensor(result[-window_size:]) # batch, window, n feature
    # inference
    x = x.flatten(1).to(device)
    with torch.inference_mode():
      pred = model(x).cpu().numpy()
    # add predictions to result
    if len_result < test_size - predicton_size:
      result = np.append(result, pred[0][0])
    else:
      result = np.concatenate([result, pred[0]])
      break
    len_result += 1
  return result
