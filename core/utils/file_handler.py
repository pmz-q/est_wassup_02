import os
from pathlib import Path
import json
import jsonpickle
from typing import Literal


ROOT_PATH = Path(__file__).parent.parent.parent

def get_root_path() -> str:
  """
  Returns:
      absolute path of "est_wassup_02"
  """
  return ROOT_PATH

def get_absolute_path(path: str) -> str:
  return os.path.abspath(path)

def get_list_of_csv_files(absolute_dir_path: str, extension: str='.csv') -> list[str]:
  csv_files = {}
  for file_name in os.listdir(absolute_dir_path):
    if file_name.count(extension):
      csv_files[file_name.replace(extension, f'_{extension}')] = file_name
  return csv_files

def get_list_of_auto_configs(config_type: Literal["arima", "nn", "tst"]) -> list[str]:
  config_files = []
  file_dir = f"{get_root_path()}/auto-configs-{config_type}"
  for file_name in os.listdir(file_dir):
    if file_name.count('.py'):
      config_files.append(f"{file_dir}/{file_name}")
  return config_files

def create_path_if_not_exists(path: str, remove_filename: bool=True, split_by: str='/') -> str:
  """
  Returns:
      str: file path
  """
  if remove_filename:
    path = get_dirs_only(path, split_by)
  if os.path.exists(path):
    path = f"{path}_1"
  os.makedirs(path, exist_ok=True)
  return path

def combine_paths(dir: str, filename: str):
  return f"{dir}/{filename}"

def get_dirs_only(path: str, split_by: str='/') -> str:
  arr = path.split(split_by)[:-1]
  return split_by.join(arr)

def save_params_json(json_path: str, object_to_save: dict, pickle: bool=True):
  with open(json_path, 'w', encoding='utf-8') as f:
    if pickle:
      object_to_save = jsonpickle.encode(object_to_save)
    json.dump(object_to_save, f, ensure_ascii=False, indent=2)