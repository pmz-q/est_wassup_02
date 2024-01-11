import os
from pathlib import Path
import json
import jsonpickle
from typing import Literal, List


ROOT_PATH = Path(__file__).parent.parent.parent

def get_root_path() -> str:
  """
  Returns:
      absolute path of "est_wassup_02"
  """
  return ROOT_PATH

def get_absolute_path(path: str) -> str:
  return os.path.abspath(path)

def get_list_of_csv_files(absolute_dir_path: str, extension: str='.csv') -> List[str]:
  csv_files = {}
  for file_name in os.listdir(absolute_dir_path):
    if file_name.count(extension):
      csv_files[file_name.replace(extension, f'_{extension}')] = file_name
  return csv_files

def get_list_of_auto_configs(config_type: Literal["arima", "nn", "tst"]) -> List[str]:
  config_files = []
  file_dir = f"{get_root_path()}/auto-configs-{config_type}"
  for file_name in os.listdir(file_dir):
    if file_name.count('.py'):
      config_files.append(f"{file_dir}/{file_name}")
  return config_files

def create_path_if_not_exists(path: str, remove_filename: bool=True, split_by: str='/', create_new_path: bool=True) -> str:
  """
  Returns:
      str: file path
  """
  if remove_filename:
    path = get_dirs_only(path, split_by)
  new_path_num = 1
  new_path = path
  while os.path.exists(new_path) and create_new_path:
    new_path = f"{path}_{new_path_num}"
    new_path_num += 1
  os.makedirs(new_path, exist_ok=True)
  return new_path

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
