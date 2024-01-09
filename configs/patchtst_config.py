from core.utils import get_root_path
from models.patch_tst import PatchTST   #### TODO 선들님께 models로 from찍으면 안 되는거 물어보기...
import torch
from torch import optim
import torch.nn.functional as F
from torch.nn import ReLU


ROOT_PATH = get_root_path()
EXPERIMENT_NAME = "patch_test_1"

TARGET_COL = "target"
MODEL = PatchTST

EPOCHS = 2
LEARNING_RATE = 0.0001
BATCH_SIZE = 32

PATCH_LENGTH = 16 # 논문 수치
N_PATCHES = 64 # 논문 수치: 64 or 96

MODEL_DIM = 128 # HIDDEN_DIM같은 것
NUM_HEADS = 8 # 논문 수치로 일단 고정
NUM_LAYERS = 3 # 논문 수치로 일단 고정

PRED_SIZE = 24
# WINDOW_SIZE = int(PATCH_LENGTH*N_PATCHES/2) # 512 약 21일

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
    "window_size": int(PATCH_LENGTH*N_PATCHES/2),
    "model": MODEL,
    "model_params": {
        "n_token": N_PATCHES,
        "input_dim": PATCH_LENGTH,  
        "model_dim": MODEL_DIM,  
        "num_heads": NUM_HEADS, 
        "num_layers": NUM_LAYERS, 
        "output_dim": PRED_SIZE
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
            "patch_length": PATCH_LENGTH,
            "n_patches": N_PATCHES,
            "prediction_length": PRED_SIZE
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
