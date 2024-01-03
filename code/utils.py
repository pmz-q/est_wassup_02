import pandas as pd
import torch

class TimeseriesDataset(torch.utils.data.Dataset):
    def __init__(self, data:pd.Series, window_size, prediction_size):
        self.data = data
        self.window_size = window_size
        self.prediction_size = prediction_size

    def __len__(self):
        return len(self.data) - self.window_size - self.prediction_size + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:(idx+self.window_size)] # 예측을 위해 볼 과거 값들
        y = self.data[(idx+self.window_size):(idx+self.window_size+self.prediction_size)]  # 예측하려는 미래 값(들)
        return x, y