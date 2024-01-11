from core.metrics import mae, mape, r2_score
from core.trains import arima
from core.utils import (
    get_args_parser,
    create_path_if_not_exists,
    save_params_json,
    combine_paths
)
import matplotlib.pyplot as plt
import pandas as pd
import joblib


def main(config):
    # get preprocessed csv files
    input_data = config.get("input_data")
    df_train = pd.read_csv(input_data.get("train_csv"), index_col=0)
    df_test = pd.read_csv(input_data.get("test_csv"), index_col=0)
    
    # applying test size
    df_test = df_test.iloc[-config.get("test_size"):]
    # change index data type
    df_train.index = pd.to_datetime(df_train.index)
    df_test.index = pd.to_datetime(df_test.index)

    # arima params
    arima_params = config.get("arima_params")

    # ARIMA or SARIMA
    arima_model = config.get("arima_model")

    pred, tst_ds = arima(df_train, df_test, arima_model, arima_params)
    tst_ds= pd.DataFrame(tst_ds)
    pred= pd.DataFrame(pred)

    # inverse scaling
    y_scaler = None
    try:
        y_scaler = joblib.load(input_data.get("y_scaler_save"))
    except:
        pass
    
    if y_scaler != None:
        tst_ds["target"] = y_scaler.inverse_transform(tst_ds["target"].values.reshape(-1, 1))
        pred["predicted_mean"] = y_scaler.inverse_transform(pred["predicted_mean"].values.reshape(-1, 1))

    # Save plot
    output_data = config.get("output_data")
    
    pred_save_path = output_data.get("pred_save_path")
    actual_save_path = output_data.get("actual_save_path")
    root_dir = create_path_if_not_exists(output_data.get("root_dir"), remove_filename=False)
    pred.to_csv(combine_paths(root_dir, pred_save_path))
    tst_ds.to_csv(combine_paths(root_dir, actual_save_path))

    plot_img_path = combine_paths(root_dir, output_data.get("plot_img_path"))

    plt.figure(figsize=(20, 10))
    plt.plot(tst_ds.index, tst_ds.values, pred.index, pred.values)
    plt.title(
        f"{arima_model}, MAPE:{mape(pred.values,tst_ds['target'].values):.4f}, MAE:{mae(pred.values,tst_ds['target'].values):.4f}, R2:{r2_score(pred.values,tst_ds['target'].values):.4f}"
    )
    plt.savefig(plot_img_path)  # save as png

    # Save parameters
    json_path = combine_paths(root_dir, output_data.get("json_path"))
    save_params_json(json_path, object_to_save=config, pickle=False)

if __name__ == "__main__":
    args = get_args_parser(config_type="train_arima").parse_args()
    config = {}
    exec(open(args.config, encoding="utf-8").read())
    main(config)
    print("finished! Please check the output folder.")