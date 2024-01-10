import pandas as pd
from .tools import (
  create_new_column_names,
  flatten_multi_index_columns
)


JOIN_BY = ['datetime', 'county', 'data_block_id']
LAT_LON_COLUMNS = ['latitude', 'longitude']
AGG_STATS = ['mean'] # 'min', 'max', 'std', 'median'

def create_historical_weather_features(
  historical_weather: pd.DataFrame, county_mapper: pd.DataFrame
) -> pd.DataFrame:
    # To datetime
    historical_weather['datetime'] = pd.to_datetime(historical_weather['datetime'])
    
    # Add county
    historical_weather[LAT_LON_COLUMNS] = historical_weather[LAT_LON_COLUMNS].astype(float).round(1)
    historical_weather = historical_weather.merge(county_mapper, how = 'left', on = LAT_LON_COLUMNS)

    # Modify column names - specify suffix
    historical_weather = create_new_column_names(
      historical_weather,
      suffix='_h',
      columns_no_change = LAT_LON_COLUMNS + JOIN_BY
    )
    
    # Group by & calculate aggregate stats 
    agg_columns = [col for col in historical_weather.columns if col not in LAT_LON_COLUMNS + JOIN_BY]
    agg_dict = {agg_col: AGG_STATS for agg_col in agg_columns}
    historical_weather = historical_weather.groupby(JOIN_BY).agg(agg_dict).reset_index() 
    
    # Flatten the multi column aggregates
    historical_weather = flatten_multi_index_columns(historical_weather) 
    
    # Test set has 1 day offset for hour<11 and 2 day offset for hour>11
    historical_weather['hour_h'] = historical_weather['datetime'].dt.hour
    historical_weather['datetime'] = (
      historical_weather
      .apply(
        lambda x: 
          x['datetime'] + pd.DateOffset(1) 
          if x['hour_h']< 11 
          else x['datetime'] + pd.DateOffset(2),
        axis=1
      )
    )
    
    return historical_weather

def merge_historical_weather(
  origin_df: pd.DataFrame,
  historical_weather_df: pd.DataFrame,
  county_mapper_df: pd.DataFrame
) -> pd.DataFrame:
  historical_weather = create_historical_weather_features(historical_weather_df, county_mapper_df)
  return origin_df.merge(historical_weather, how='left', on=JOIN_BY)

