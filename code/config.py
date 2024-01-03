import torch
import torch.nn.functional as F
import torchmetrics
from models import ANNMulti


# preprocessed csv(train, test)가 있다고 가정한 상태에서 진행합니다. 
config = {

  'files': {
    'train_csv': '../data/tester/train.csv',
    'test_csv': '../data/tester/test.csv',
    'model_output': '../output/model_output/model.pth',  # 구분할 수 있도록 설정하기
    'pred_output': '../output/pred_output/pred.csv', # 구분할 수 있도록 설정하기
    'result_image': '../output/results/config_name.png', 
  },

  'model': ANNMulti,
  'model_params': {
    'input_dim': 9,
    'output_dim': 1,
    'hidden_dim': 128,
    'input_channel': 2,
    'activation': F.relu,
  },

  'train_params': {
    'data_loader_params': {
      'batch_size': 32,
      'shuffle': True,
    },
    'dataset_params': {
      'window_size': 9,
      'prediction_size': 1
    },
    'loss': F.mse_loss,
    'optim': torch.optim.Adam,
    'optim_params': {
      'lr': 0.0001,
    },
    'metric': F.l1_loss, # metric collection?으로 여러개 받아오기
    'device': 'cpu',  #  cuda - gpu 
    'epochs': 1000,
  },

  'arima_params':{
      'target': "SUNACTIVITY",
      'order': {
          'p': 9,
          'd': 0,
          'q': 1
      },
      'season_order': {
          'P': 0,
          'D': 0,
          'Q': 0,
          's': 0, # seasonal freq 
      },
  }
}