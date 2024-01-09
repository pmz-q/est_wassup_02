from core.utils import get_root_path


ROOT_PATH = get_root_path()
EXPERIMENT_NAME = "arima_test_1"

ARIMA_MODEL = "ARIMA" # ARIMA or SARIMA
ORDER = (25, 1, 0) # p, d, q
SEASON_ORDER = (0, 0, 0, 24) # P, D, Q, S
TREND = "c"
FREQ = "h"

config = {
  "input_data": {
    "train_csv": f"{ROOT_PATH}/features/train_X.csv",
    "test_csv": f"{ROOT_PATH}/features/test_X.csv",
  },
  "output_data": {
    "plot_img_path": f"{ROOT_PATH}/output/{EXPERIMENT_NAME}/output.png",
    "json_path": f"{ROOT_PATH}/output/{EXPERIMENT_NAME}/output.json",
    "pred_save_path": f"{ROOT_PATH}/output/{EXPERIMENT_NAME}/pred_save.csv",
    "actual_save_path": f"{ROOT_PATH}/output/{EXPERIMENT_NAME}/actual_save.csv",
  },
  "arima_model": ARIMA_MODEL,
  "arima_params":{
    "order": ORDER,
    "seasonal_order": SEASON_ORDER,
    "trend": TREND,
    "freq": FREQ
  }
}