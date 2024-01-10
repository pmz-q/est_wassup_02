import datetime
import holidays # https://github.com/vacanza/python-holidays
import pandas as pd


def merge_holidays(
  origin_df: pd.DataFrame
) -> pd.DataFrame:
  estonian_holidays = holidays.country_holidays('EE', years=range(2021, 2026))
  estonian_holidays = list(estonian_holidays.keys())

  origin_df['country_holiday'] = origin_df.apply(lambda row: (datetime.date(row['year'], row['month'], row['day']) in estonian_holidays) * 1, axis=1)
  
  return origin_df