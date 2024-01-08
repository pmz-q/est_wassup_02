import torch
import numpy as np


class PatchTSTDataset(torch.utils.data.Dataset):
  def __init__(self, ts:np.array, patch_size:int, n_patch:int, n_token:int):
    self.patch_size = patch_size
    self.n_patch = n_patch
    self.n_token = n_token
    self.window_size = int(patch_size * n_patch * n_token / 2)
    self.forecast_size = patch_size
    self.data = ts

  def __len__(self):
    return len(self.data) - self.window_size - self.forecast_size + 1

  def __getitem__(self, idx:int):
    look_back = self.data[idx:(idx+self.window_size)]
    look_back = np.concatenate([look_back] + [look_back[-self.patch_size:]] * int(self.n_patch / 2))
    x = np.array([look_back[i*int(self.patch_size*self.n_patch/2):(i+2)*int(self.patch_size*self.n_patch/2)] for i in range(self.n_token)])

    y = self.data[(idx+self.window_size):(idx+self.window_size+self.forecast_size)]
    return x, y