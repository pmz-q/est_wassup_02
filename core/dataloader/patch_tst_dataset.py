import torch
import numpy as np


class PatchTSTDataset(torch.utils.data.Dataset):
  def __init__(self, ts:np.array, patch_length:int=16, n_patches:int=6, prediction_length:int=4):
    self.P = patch_length
    self.N = n_patches
    self.L = int(patch_length * n_patches / 2)  # look-back window length = 512
    self.T = prediction_length
    self.data = ts

  def __len__(self):
    return len(self.data) - self.L - self.T + 1
    # return len(self.data) - self.L

  def __getitem__(self, i):
    look_back = self.data[i:(i+self.L)] # data[0:512] -> (512,)
    look_back = np.concatenate([look_back, look_back[-1]*np.ones(int(self.P / 2), dtype=np.float32)])
    # print((look_back[-1]*np.ones(int(self.P / 2))).shape) #(마지막걸로 상수처럼 채우기) (8,)
    x = np.array([look_back[i*int(self.P/2):(i+2)*int(self.P/2)] for i in range(self.N)])
    y = self.data[(i+self.L):(i+self.L+self.T)]
    return x, y