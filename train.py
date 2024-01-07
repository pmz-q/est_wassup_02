from core.dataloader import TimeseriesDataset
from core.metrics import mae, mape, r2_score
from core.trains import nn_train
from core.utils import create_path_if_not_exists, save_params_json, get_args_parser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import joblib


# train & test together
def main(cfg):
    # get preprocessed csv files
    input_data = cfg.get("input_data")
    train_data = pd.read_csv(input_data.get('train_csv'), index_col=0)
    test_data = pd.read_csv(input_data.get('test_csv'), index_col=0)
    
    # input channel - num of features
    n_featues = train_data.shape[1]
    
    # change index data type
    train_data.index = pd.to_datetime(train_data.index)
    test_data.index = pd.to_datetime(test_data.index)
    
    trn_ds = train_data[train_data.index < test_data.index[0]].to_numpy(dtype=np.float32)
    tst_ds = train_data[train_data.index >= test_data.index[0]].to_numpy(dtype=np.float32)
    
    # Data Loader
    train_params = cfg.get("train_params")
    # train
    ds_params = train_params.get("dataset_params")
    dl_params = train_params.get("data_loader_params")
    trn_ds = TimeseriesDataset(trn_ds, **ds_params)
    trn_dl = DataLoader(trn_ds, **dl_params)
    # test
    tst_ds = TimeseriesDataset(tst_ds, **ds_params)
    tst_dl = DataLoader(tst_ds, shuffle=False, batch_size=len(tst_ds))
    
    
    # Model
    model_params = cfg.get("model_params")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model = cfg.get("model")
    model_params = cfg.get("model_params")
    model = Model(**{**model_params, "input_channel": n_featues}).to(device)
    
    # Optim
    Optim = train_params.get("optim")
    optim_params = train_params.get("optim_params")
    optimizer = Optim(model.parameters(), **optim_params)

    loss = train_params.get("loss") # Loss function 
    metric = train_params.get("metric")
    epochs = train_params.get("epochs")
    
    # TRAIN
    trn_loss_lst, tst_loss_lst, pred = nn_train(epochs, trn_dl, tst_dl, model, loss, metric, optimizer, device)

    # Save pre-trained weight  -> 재현 용 (저장은 해두기)
    output_data = cfg.get("output_data")
    create_path_if_not_exists(output_data.get("output_train"))
    torch.save(model.state_dict(), output_data.get("output_train"))

    # Prediction
    x, y = next(iter(tst_dl)) # to retrieve x, y from tst_dl
    y = y[:,0].unsqueeze(1).to(device)  
    # Inverse scaling
    y_scaler = None
    try:
        y_scaler = joblib.load(input_data.get("y_scaler_save"))
    except:
        pass
    
    if y_scaler != None:
        y = y_scaler.inverse_transform(y.cpu())
        p = y_scaler.inverse_transform(pred)
        # y = y/scaler.scale_[0] + scaler.min_[0] # inverse scaling
        # p = pred/scaler.scale_[0] + scaler.min_[0] # inverse scaling

    y = np.concatenate([y[:,0], y[-1,1:]]) # true 
    p = np.concatenate([p[:,0], p[-1,1:]]) # prediction

    # print(p)
    pred_df = pd.DataFrame({'prediction': p})
    pred_df.to_csv(output_data.get("pred_output"))
    # TODO: submission 파일에 pred 넣기

    ### Visualization with Results ###
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 8))
 
    # Plotting the losses
    ax1.plot(range(epochs), trn_loss_lst, label='Training Loss')
    ax1.plot(range(epochs), tst_loss_lst, label='Test Loss')
    ax1.set_title(f"{Model}, last train loss: {trn_loss_lst[-1]:.6f}, last test loss: {tst_loss_lst[-1]:.6f}")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Visualize results - true, pred with metrics
    ax2.plot(test_data.index[ds_params["window_size"]:], y, label="True")
    ax2.plot(test_data.index[ds_params["window_size"]:], p, label="Prediction")
    ax2.set_title(f"{Model}, MAPE:{mape(p,y):.4f}, MAE:{mae(p,y):.4f}, R2:{r2_score(p,y):.4f}")
    ax2.legend()
    plt.xticks(test_data.index[ds_params["window_size"]::3])
    
    plt.tight_layout()
    plt.savefig(output_data.get("plot_img_path")) # save as png
    
    save_params_json(output_data.get("json_path"), cfg)

if __name__ == "__main__":
    args = get_args_parser(config_type='train').parse_args()
    config = {}
    exec(open(args.config).read())
    main(config)
    print("finished! Please check the output folder.")