from core.utils import get_root_path


ROOT_PATH = get_root_path()
EXPERIMENT_NAME = "arima_test_1"

ARIMA_MODEL = "SARIMA" # ARIMA or SARIMA
ORDER = (24, 0, 24) # p, d, q
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
  },
  "arima_model": ARIMA_MODEL,
  "arima_params":{
    "order": ORDER,
    "seasonal_order": SEASON_ORDER,
    "trend": TREND,
    "freq": FREQ
  }
}