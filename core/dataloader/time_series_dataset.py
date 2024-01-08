import pandas as pd
import torch

class TimeseriesDataset(torch.utils.data.Dataset):
    def __init__(self, data:pd.DataFrame, window_size:int, prediction_size:int, target_col:str):
        self.data = data
        self.target_col = target_col
        self.window_size = window_size # 앞에 몇개를 볼 것인가
        self.prediction_size = prediction_size # 몇개를 예측할 것인가

    def __len__(self):
        return len(self.data) - self.window_size - self.prediction_size + 1
    
    def __getitem__(self, idx:int):
        x = self.data[idx:(idx+self.window_size)] # 예측을 위해 볼 과거 값들
        y = self.data[(idx+self.window_size):(idx+self.window_size+self.prediction_size)][0]  # 예측하려는 미래 값(들)
        return x, y