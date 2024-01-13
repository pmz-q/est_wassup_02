from pathlib import Path
from typing import Literal
import argparse


ROOT_PATH = Path(__file__).parent.parent.parent

CONFIG_PATH_MAPPER = {
  "preprocess": f"{ROOT_PATH}/configs/preprocess_config.py",
  "train_arima": f"{ROOT_PATH}/configs/arima_config.py",
  "train": f"{ROOT_PATH}/configs/train_config.py",
  "train_patchtst": f"{ROOT_PATH}/configs/patchtst_config.py",
  "train_patchtst_origin": f"{ROOT_PATH}/configs/patchtst_origin_config.py",
}

DESC_TITLE_MAPPER = {
  "preprocess": "Preprocessing: Generate Dataset ",
  "train_arima": "Train: ARIMA & SARIMAX ",
  "train": "Train: NN",
  "train_patchtst": "Train: PatchTST",
  "train_patchtst_origin": "Train: PatchTST: Origin"
}

def get_args_parser(
  add_help=True,
  config_type: Literal["preprocess", "train_arima", "train", "train_patchtst", "train_patchtst_origin"]="preprocess"
):
  parser = argparse.ArgumentParser(description=DESC_TITLE_MAPPER[config_type], add_help=add_help)
  parser.add_argument("-c", "--config", default=CONFIG_PATH_MAPPER[config_type], type=str, help="configuration file")
  return parser
