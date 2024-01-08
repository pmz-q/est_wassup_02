from core.utils import get_root_path
from models import ANN
import torch
from torch import optim
import torch.nn.functional as F
from torch.nn import ReLU


ROOT_PATH = get_root_path()
EXPERIMENT_NAME = "ann_test_1"

TARGET_COL = "target"
MODEL = ANN

EPOCHS = 5
LEARNING_RATE = 0.001
BATCH_SIZE = 32

PRED_SIZE = 10
WINDOW_SIZE = 24
ACTIVATION_FUNC = ReLU()
HIDDEN_DIM = [128, 64]
USE_DROP = True
DROP_RATIO = 0.3

# TODO: scheduler 로직은 미구현 상태
USE_SCHEDULER = True
SCHEDULER = optim.lr_scheduler.CosineAnnealingLR
SCHEDULER_PARAMS = {
    "T_max": 10,
    "eta_min": 0.0001,
}

LOSS_FUNC = F.mse_loss
OPTIM = torch.optim.Adam
METRIC = F.l1_loss


config = {
    "input_data": {
        "train_csv": f"{ROOT_PATH}/features/train_X.csv",
        "test_csv": f"{ROOT_PATH}/features/test_X.csv",
        "y_scaler_save": f"{ROOT_PATH}/features/y_scaler.save",
    },
    "output_data": {
        "output_train": f"{ROOT_PATH}/output/{EXPERIMENT_NAME}/model.pth",
        "plot_img_path": f"{ROOT_PATH}/output/{EXPERIMENT_NAME}/output.png",
        "json_path": f"{ROOT_PATH}/output/{EXPERIMENT_NAME}/config.json",
        "output_pred": f"{ROOT_PATH}/output/{EXPERIMENT_NAME}/pred.csv",
    },
    "model": MODEL,
    "model_params": {
        "input_dim": WINDOW_SIZE,  # window size
        "input_channel": "auto",  # number of features
        "hidden_dim": HIDDEN_DIM,
        "use_drop": USE_DROP,
        "drop_ratio": DROP_RATIO,
        "activation": ACTIVATION_FUNC,
    },
    "train_params": {
        "use_scheduler": USE_SCHEDULER,
        "scheduler_cls": SCHEDULER,
        "scheduler_params": SCHEDULER_PARAMS,
        "loss": LOSS_FUNC,
        "optim": OPTIM,
        "metric": METRIC,
        "epochs": EPOCHS,
        "dataset_params": {
            "window_size": WINDOW_SIZE,
            "prediction_size": PRED_SIZE,
            "target_col": TARGET_COL,
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
