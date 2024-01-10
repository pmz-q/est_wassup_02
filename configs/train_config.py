from core.utils import get_root_path
from models import ANN
import torch
from torch import optim
import torch.nn.functional as F
from torch.nn import ReLU


ROOT_PATH = get_root_path()
EXPERIMENT_NAME = "ann_test_1"

TEST_SIZE = 24

# train params
EPOCHS = 40
LEARNING_RATE = 0.00001
BATCH_SIZE = 32
LOSS_FUNC = F.mse_loss
OPTIM = torch.optim.Adam
METRIC = F.l1_loss

# nn params
MODEL = ANN
PRED_SIZE = 24
WINDOW_SIZE = 12
ACTIVATION_FUNC = ReLU()
HIDDEN_DIM = [128, 64]
USE_DROP = True
DROP_RATIO = 0.3
OUTPUT_FUNC = F.sigmoid # Y scaler 에 따라서 변경되어야 함
# y scaler 가 None 인 경우, lambda x: x 이렇게 넣어주면 됨.

# scheduler
USE_SCHEDULER = False
SCHEDULER = optim.lr_scheduler.CosineAnnealingLR
SCHEDULER_PARAMS = {
    "T_max": 50,
    "eta_min": 0.000001,
}

# early stop
USE_EARLY_STOP = True
EARLY_STOP_PARAMS = {
    "patience_limit": 20 # 몇 번의 epoch까지 지켜볼지를 결정
}

config = {
    "input_data": {
        "train_csv": f"{ROOT_PATH}/features/train_X.csv",
        # "test_csv": f"{ROOT_PATH}/features/test_X.csv",
        "y_scaler_save": f"{ROOT_PATH}/features/y_scaler.save",
    },
    "output_data": {
        "output_train": f"{ROOT_PATH}/output/{EXPERIMENT_NAME}/model.pth",
        "plot_img_path": f"{ROOT_PATH}/output/{EXPERIMENT_NAME}/output.png",
        "json_path": f"{ROOT_PATH}/output/{EXPERIMENT_NAME}/config.json",
        "output_pred": f"{ROOT_PATH}/output/{EXPERIMENT_NAME}/pred.csv",
    },
    "test_size": TEST_SIZE,
    "model": MODEL,
    "model_params": {
        "input_dim": WINDOW_SIZE,  # window size
        "input_channel": "auto",  # number of features
        "hidden_dim": HIDDEN_DIM,
        "use_drop": USE_DROP,
        "drop_ratio": DROP_RATIO,
        "activation": ACTIVATION_FUNC,
        "output_dim": PRED_SIZE,
        "output_func": OUTPUT_FUNC
    },
    "train_params": {
        "use_scheduler": USE_SCHEDULER,
        "scheduler_cls": SCHEDULER,
        "scheduler_params": SCHEDULER_PARAMS,
        "use_early_stop": USE_EARLY_STOP,
        "early_stop_params": EARLY_STOP_PARAMS,
        "loss": LOSS_FUNC,
        "optim": OPTIM,
        "metric": METRIC,
        "epochs": EPOCHS,
        "dataset_params": {
            "window_size": WINDOW_SIZE,
            "prediction_size": PRED_SIZE,
        },
        "data_loader_params": {
            "batch_size": BATCH_SIZE,
            "shuffle": True,
        },
        "optim_params": {
            "lr": LEARNING_RATE,
        },
    },
}
