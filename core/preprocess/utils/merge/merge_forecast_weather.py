import pandas as pd
from .tools import (
  create_new_column_names,
  flatten_multi_index_columns
)


JOIN_BY = ['datetime', 'county', 'data_block_id']
LAT_LON_COLUMNS = ['latitude', 'longitude']
AGG_STATS = ['mean'] # 'min', 'max', 'std', 'median'

def create_forecast_weather_features(
  forecast_weather: pd.DataFrame, county_mapper: pd.DataFrame
) -> pd.DataFrame:
    # Rename column and drop
    forecast_weather = (
      forecast_weather
      .rename(columns = {'forecast_datetime': 'datetime'})
      .drop(columns = 'origin_datetime') # not needed
    )
    
    # To datetime
    forecast_weather['datetime'] = (
      pd.to_datetime(forecast_weather['datetime'])
      .dt
      .tz_localize(None)
      )

    # Add county
    forecast_weather[LAT_LON_COLUMNS] = forecast_weather[LAT_LON_COLUMNS].astype(float).round(1)
    forecast_weather = forecast_weather.merge(county_mapper, how='left', on=LAT_LON_COLUMNS)
    
    # Modify column names - specify suffix
    forecast_weather = create_new_column_names(
      forecast_weather,
      suffix='_f',
      columns_no_change = LAT_LON_COLUMNS + JOIN_BY
    )
    
    # Group by & calculate aggregate stats 
    agg_columns = [col for col in forecast_weather.columns if col not in LAT_LON_COLUMNS + JOIN_BY]
    agg_dict = {agg_col: AGG_STATS for agg_col in agg_columns}
    forecast_weather = forecast_weather.groupby(JOIN_BY).agg(agg_dict).reset_index() 
    
    # Flatten the multi column aggregates
    forecast_weather = flatten_multi_index_columns(forecast_weather)     
    return forecast_weather
  
def merge_forecast_weather(
  origin_df: pd.DataFrame,
  forecast_weather_df: pd.DataFrame,
  county_mapper_df: pd.DataFrame
) -> pd.DataFrame:
  forecast_weather = create_forecast_weather_features(forecast_weather_df, county_mapper_df)
  return origin_df.merge(forecast_weather, how='left', on=JOIN_BY)
