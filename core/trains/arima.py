from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from typing import Literal
import pandas as pd


def arima(
  trn_ds: pd.DataFrame, test_size: int, 
  predict_start_date: str, predict_end_date: str,
  model_type: Literal["ARIMA", "SARIMA"], arima_parmas: dict
):
  """  
  Returns: pred
  """
  trn_ds.index.freq = trn_ds.index.inferred_freq

  if model_type == "SARIMA":
    model = SARIMAX(trn_ds, **arima_parmas)
    pred = model.fit(disp=False).forecast(test_size)
  elif model_type == "ARIMA":
    model = ARIMA(trn_ds, order=arima_parmas.get("order"))
    pred = model.fit().predict(predict_start_date, predict_end_date, dynamic=True)
  
  return pred
