import pandas as pd
from typing import Literal
PERIOD = 24

# 24 rows before data + 24 rows after data => mean
"""
def custom_mean(df: pd.DataFrame):
    # print("in the test mean", df['data_block_id'][0], df['hour'][0], df['data_block_id'][PERIOD], df['hour'][PERIOD])
    # print(df.columns)
    for column in df.columns:
        missing_rows = df[column].isnull()
        missing_indices = df[missing_rows].index
        
        for index in missing_indices:
            if index >= PERIOD:
                before_value = df.iloc[index - PERIOD][column]
                after_value = df.iloc[index + PERIOD][column]
                mean_value = (before_value + after_value) / 2
                df.at[index, column] = mean_value
    return df
"""
def custom_mean(df: pd.DataFrame):
    for column in df.columns:
        missing_rows = df[column].isnull()
        missing_indices = df[missing_rows].index
        
        for index in missing_indices:
            calculated_index = index + PERIOD
            if 0 <= calculated_index < len(df):
                before_value = df.iloc[index - PERIOD][column]
                after_value = df.iloc[calculated_index][column]
                mean_value = (before_value + after_value) / 2
                df.at[index, column] = mean_value
    return df


def fill_num_strategy(strategy: Literal["min", "max", "mean", "custom_mean"]) -> callable:
  def fill_num(df_num: pd.DataFrame) -> None:
    df = df_num.copy()
    if strategy == "custom_mean":
        df_num = custom_mean(df)
    else:
        df.fillna(0, inplace=True)
        if strategy == "mean":
            fill_values = df.mean(axis=0)
        elif strategy == "min":
            fill_values = df.min(axis=0)
        elif strategy == "max":
            fill_values = df.max(axis=0)
        df_num = df_num.fillna(fill_values)
    return df_num
  return fill_num