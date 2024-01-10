import pandas as pd
from .merge import (
  merge_client,
  merge_electricity_prices,
  merge_gas_prices,
  merge_forecast_weather,
  merge_historical_weather,
  merge_holidays
)


def create_data_features(X_df: pd.DataFrame):
  '''📊Create features for main data (test or train) set📊'''
  # To datetime
  if 'prediction_datetime' in X_df.columns:
    X_df.rename(columns={'prediction_datetime': 'datetime'}, inplace=True)
  X_df['datetime'] = pd.to_datetime(X_df['datetime'])
  
  # Time period features
  X_df['date'] = X_df['datetime'].dt.normalize()
  X_df['day'] = X_df['datetime'].dt.day
  X_df['year'] = X_df['datetime'].dt.year
  X_df['quarter'] = X_df['datetime'].dt.quarter
  X_df['month'] = X_df['datetime'].dt.month
  X_df['week'] = X_df['datetime'].dt.isocalendar().week
  X_df['hour'] = X_df['datetime'].dt.hour
  
  # Day features
  X_df['day_of_year'] = X_df['datetime'].dt.day_of_year
  X_df['day_of_month']  = X_df['datetime'].dt.day
  X_df['day_of_week'] = X_df['datetime'].dt.day_of_week
  return X_df

def merge(X_df: pd.DataFrame, add_data: dict, county_mapper: pd.DataFrame) -> pd.DataFrame:
  merge_function = {
    "client": merge_client,
    "historical_weather": merge_historical_weather,
    "forecast_weather": merge_forecast_weather,
    "electricity_prices": merge_electricity_prices,
    "gas_prices": merge_gas_prices
  }
  
  # Create features for relevant dataset
  df = create_data_features(X_df)
  for csv_name, csv_df in add_data.items():
    try:
      if csv_name in ["historical_weather", "forecast_weather"]:
        df = merge_function[csv_name](df, csv_df, county_mapper.iloc[:,1:].copy())
      else:
        df = merge_function[csv_name](df, csv_df)
    except KeyError:
      print(f"[merge_dataset.py]-{csv_name} function unavailable.")
  
  df = merge_holidays(df)
  
  return df

def merge_dataset(X_df_train: pd.DataFrame, X_df_test: pd.DataFrame, add_data: dict):
  """
  Return: (merged_train_df, merged_test_df)
  """
  merged_train_df = merge(X_df_train, add_data['train'], add_data['county_mapper']['county_mapper'])
  merged_test_df = merge(X_df_test, add_data['test'], add_data['county_mapper']['county_mapper'])

  return (merged_train_df, merged_test_df)
