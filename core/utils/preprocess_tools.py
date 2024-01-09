import pandas as pd
from typing import Literal


def fill_num_strategy(strategy: Literal["min", "max", "mean", "custom_mean"]) -> callable:
  def fill_num(df_num: pd.DataFrame) -> None:
    df = df_num.copy()
    df.fillna(0)
    if strategy == "mean":
        fill_values = df.mean(axis=0)
    elif strategy == "min":
        fill_values = df.min(axis=0)
    elif strategy == "max":
        fill_values = df.max(axis=0)
    df_num = df_num.fillna(fill_values)
    return df_num
  return fill_num