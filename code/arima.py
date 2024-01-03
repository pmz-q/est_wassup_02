from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd 
import numpy as np 

def arima(cfg):
    arima_params = cfg.get("arima_params")
    order = arima_params.get("order")
    s_order = arima_params.get("season_order")

    # get preprocessed csv files
    files = cfg.get("files")
    train_df = pd.read_csv(files.get('train_csv'), index_col=0)
    test_df = pd.read_csv(files.get('test_csv'), index_col=0)

    tgt = arima_params.get("target")
    trn_ds = train_df[tgt]
    tst_ds = test_df[tgt]

    model = SARIMAX(trn_ds, order = (order['p'], order['d'], order['q']), 
                    seasonal_order= (s_order['P'], s_order['D'], s_order['Q'], s_order['s']))
    pred = model.fit().forecast(len(tst_ds))
    pred = pd.Series(pred, index = tst_ds.index)
    print(pred)

    
    
# def arima(train_df, test_df, target: str="SUNACTIVITY", 
#           p:int=0,d:int=0,q:int=0, pred_start:int=1989, pred_end:int=2008):
#     trn_ds = train_df.target[:-20]  #TODO test_df에서 test length 추출해서 사용
#     tst_ds = train_df.target[-20:]  #TODO test_df 통으로 사용
#     model = ARIMA(trn_ds, order = (p,d,q)).fit()
#     pred_start = 
#     pred = model.predict(pred_start, pred_end, dynamic = True)

#     return pred


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






