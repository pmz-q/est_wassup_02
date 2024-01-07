from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from typing import Literal
import pandas as pd


def arima(
  df_train: pd.DataFrame, df_test: pd.DataFrame,
  model_type: Literal["AIRMA", "SARIMA"], arima_parmas: dict
):
  """  
  Returns: (pred, tst_ds)
  """
  trn_ds = df_train[df_train.index < df_test.index[0]]
  tst_ds = df_train[df_train.index >= df_test.index[0]]
  trn_ds.index.freq = trn_ds.index.inferred_freq
  tst_ds.index.freq = tst_ds.index.inferred_freq  

  if model_type == "SARIMA":
    model = SARIMAX(trn_ds, **arima_parmas)
    pred = model.fit(disp=False).forecast(len(tst_ds))
  elif model_type == "AIRMA":
    model = ARIMA(trn_ds, order=arima_parmas.get("order"))
    pred = model.fit().predict(tst_ds.index[0], tst_ds.index[-1], dynamic=True)
  
  return pred, tst_ds
