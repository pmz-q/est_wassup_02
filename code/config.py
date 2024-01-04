import torch
import torch.nn.functional as F
import torchmetrics
from models import ANNMulti

# ROOT_PATH = '..'

# ROOT_DATA_PATH = f'{ROOT_PATH}/data/origin/'
# TEST_DATA_PATH = f'{ROOT_DATA_PATH}test/'
# TRAIN_DATA_PATH = f'{ROOT_DATA_PATH}train/'

# ROOT_FEATURE_PATH = f'{ROOT_PATH}/data/features'

config_name = "240104_01"
# preprocessed csv(train, test)가 있다고 가정한 상태에서 진행합니다. 
config = {

  'files': {
    'train_csv': '../data/tester/train.csv',
    'test_csv': '../data/tester/test.csv',
    'model_output': f'../output/model_output/{config_name}.pth',  # saved model weights
    'pred_output': f'../output/pred_output/{config_name}.csv', # neural network predictions
    'result_image': f'../output/results/{config_name}.png', # neural network results
    'arima_image': f'../output/results/{config_name}.png' # arima model results
  },

  'model': ANNMulti,
  'model_params': {
    'input_dim': 9, # window size
    'output_dim': 1, # prediction size
    'hidden_dim': 128,
    'input_channel': 2, # number of features 
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
  'apply_scaler': True, # Scaler # TODO scaler 종류 추가 

  'arima_model': "SARIMA", # ARIMA or SARIMA
  'arima_params':{
      'target': "SUNACTIVITY", # column name of the target
      'order': {
          'p': 9,
          'd': 0,
          'q': 0
      },
      'season_order': {
          'P': 0,
          'D': 0,
          'Q': 0,
          's': 0, # seasonal freq 
      },
  }
}