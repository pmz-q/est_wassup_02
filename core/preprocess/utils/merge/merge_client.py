import pandas as pd
from .tools import create_new_column_names


JOIN_BY = ['county', 'is_business', 'product_type', 'data_block_id']

def create_client_features(client: pd.DataFrame):
  # Modify column names - specify suffix
  client = create_new_column_names(
    client, 
    suffix='_client',
    columns_no_change = JOIN_BY
    )       
  return client

def merge_client(
  origin_df: pd.DataFrame,
  client_df: pd.DataFrame
) -> pd.DataFrame:
  client = create_client_features(client_df)
  return origin_df.merge(client, how='left', on=JOIN_BY)
