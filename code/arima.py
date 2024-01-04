from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd 
import numpy as np 
from metrics import mae, mape, r2_score
import matplotlib.pyplot as plt 

def arima(cfg):
    arima_params = cfg.get("arima_params")
    order = arima_params.get("order")
    s_order = arima_params.get("season_order")

    # get preprocessed csv files
    files = cfg.get("files")
    train_df = pd.read_csv(files.get('train_csv'), index_col=0)
    test_df = pd.read_csv(files.get('test_csv'), index_col=0)

    train_df.index = pd.to_datetime(train_df.index)
    test_df.index = pd.to_datetime(test_df.index)

    tgt = arima_params.get("target")
    trn_ds = train_df[tgt]
    tst_ds = test_df[tgt]
    trn_ds.index.freq = trn_ds.index.inferred_freq
    tst_ds.index.freq = tst_ds.index.inferred_freq  

    model_type = cfg.get("arima_model")
    if model_type == "SARIMA":
        model = SARIMAX(trn_ds, order = (order['p'], order['d'], order['q']), 
                        seasonal_order= (s_order['P'], s_order['D'], s_order['Q'], s_order['s']))
        pred = model.fit(disp=False).forecast(len(tst_ds))
    else: # "ARIMA"
        model = ARIMA(trn_ds, order = (order['p'], order['d'], order['q']))
        pred = model.fit().predict(tst_ds.index[0], tst_ds.index[-1], dynamic=True)
    
    # Plot
    pred.plot()
    tst_ds.plot(label="real")
    plt.title(f"{model_type}, MAPE:{mape(pred.values,tst_ds.values):.4f}, MAE:{mae(pred.values,tst_ds.values):.4f}, R2:{r2_score(pred.values,tst_ds.values):.4f}")
    plt.savefig(files.get("arima_image")) # save as png

### parser
def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Pytorch Model Trainer", add_help=add_help)
    parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    exec(open(args.config).read())
    arima(config)
    print("finished! Please check the output folder.")






