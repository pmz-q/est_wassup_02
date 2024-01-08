from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(kw_only=True)
class Dataset:
  train_csv: str
  test_csv: str
  add_data: dict
  index_col: str
  tst_index_col: str
  target_col: str
  use_arima: bool
  row_query: dict
  use_drop_cols: bool
  drop_cols: list[str]
  use_cols: list[str]

  def _filter_rows(self, df):
    for k, v in self.row_query.items():
      df = df[df[k] == v]
    return df
  
  def _read_df(self, split: Literal["train", "test", "add"] = "train"):
    if split == "train":
      df = pd.read_csv(self.train_csv)
      df = self._filter_rows(df)
      return df
    elif split == "test":
      df = pd.read_csv(self.test_csv)
      df.rename(columns={self.tst_index_col: self.index_col}, inplace=True)
      df = self._filter_rows(df)
      return df
    elif split == "add":
      df_add = {
        k: { 
          csv_name: pd.read_csv(csv_path)
          for csv_name, csv_path in v.items()
         } for k, v in self.add_data.items()
      }
      return df_add
    raise ValueError(f'"{split}" is not acceptable.')

  def _get_dataset(self):
    """
    Returns:
      (trn_df: DataFrame, tst_df: DataFrame, add_data: dict)
    """
    trn_df = self._read_df("train")
    tst_df = self._read_df("test")
    add_data = self._read_df("add")
    
    if self.use_arima:
        trn_df = trn_df[[self.index_col, self.target_col]]
        tst_df = tst_df[[self.index_col]]
    elif self.use_drop_cols:
      # drop `drop_cols`
      trn_df.drop(self.drop_cols, axis=1, errors='ignore', inplace=True)
      tst_df.drop(self.drop_cols, axis=1, errors='ignore', inplace=True)
    else:
      trn_df = trn_df[self.use_cols]
      tst_df = tst_df[self.use_cols]

    return (trn_df, tst_df, add_data)
