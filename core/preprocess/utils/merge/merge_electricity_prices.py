import pandas as pd
from .tools import create_new_column_names


JOIN_BY = ['datetime', 'data_block_id']

def create_electricity_features(electricity: pd.DataFrame) -> pd.DataFrame:
  # To datetime
  electricity['forecast_date'] = pd.to_datetime(electricity['forecast_date'])
  
  # Test set has 1 day offset
  electricity['datetime'] = electricity['forecast_date'] + pd.DateOffset(1)
  
  # Modify column names - specify suffix
  electricity = create_new_column_names(
    electricity, 
    suffix='_electricity',
    columns_no_change=JOIN_BY
  )
  
  return electricity

def merge_electricity_prices(
  origin_df: pd.DataFrame,
  electricity_prices_df: pd.DataFrame
) -> pd.DataFrame:
  electricity_prices = create_electricity_features(electricity_prices_df)
  return origin_df.merge(electricity_prices, how='left', on=JOIN_BY)
