from core.metrics import mae, mape, r2_score
from core.trains import arima
from core.utils import get_args_parser, create_path_if_not_exists, save_params_json
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    args = get_args_parser(config_type="train_arima").parse_args()
    config = {}
    exec(open(args.config, encoding="utf-8").read())

    # get preprocessed csv files
    input_data = config.get("input_data")
    df_train = pd.read_csv(input_data.get("train_csv"), index_col=0)
    df_test = pd.read_csv(input_data.get("test_csv"), index_col=0)

    # change index data type
    df_train.index = pd.to_datetime(df_train.index)
    df_test.index = pd.to_datetime(df_test.index)

    # arima params
    arima_params = config.get("arima_params")

    # ARIMA or SARIMA
    arima_model = config.get("arima_model")

    pred, tst_ds = arima(df_train, df_test, arima_model, arima_params)

    # Save plot
    output_data = config.get("output_data")
    plot_img_path = output_data.get("plot_img_path")
    create_path_if_not_exists(plot_img_path)

    plt.figure(figsize=(20, 10))
    plt.plot(pred.index, pred.values, tst_ds.index, tst_ds.values)
    plt.title(
        f"{arima_model}, MAPE:{mape(pred.values,tst_ds.values):.4f}, MAE:{mae(pred.values,tst_ds.values):.4f}, R2:{r2_score(pred.values,tst_ds.values):.4f}"
    )
    plt.savefig(plot_img_path)  # save as png

    # Save parameters
    json_path = output_data.get("json_path")
    save_params_json(json_path, object_to_save=config, pickle=False)
