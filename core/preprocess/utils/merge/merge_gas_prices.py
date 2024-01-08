import pandas as pd
from .tools import create_new_column_names


JOIN_BY = ['data_block_id']

def create_gas_features(gas):
  # Mean gas price
  gas['mean_price_per_mwh'] = (gas['lowest_price_per_mwh'] + gas['highest_price_per_mwh'])/2
  
  # Modify column names - specify suffix
  gas = create_new_column_names(
    gas, 
    suffix='_gas',
    columns_no_change = JOIN_BY
  )       
  return gas

def merge_gas_prices(
  origin_df: pd.DataFrame,
  gas_prices_df: pd.DataFrame
) -> pd.DataFrame:
  gas_prices = create_gas_features(gas_prices_df)
  return origin_df.merge(gas_prices, how='left', on=JOIN_BY)
