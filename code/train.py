import pandas as pd 
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics
from utils import TimeseriesDataset


# train function for one epoch
def train_one_epoch(dataloader, model, loss_func, optimizer, device):
    model.train()
    total_loss = 0. 
    for x, y in dataloader:
        x, y = x.flatten(1).to(device), y[:,:,0].to(device) # 2D로 맞춰줌
        # TODO enefit에서는 y[] index 맞춰줄때 어떤 column을 타겟으로 하는지 확인 필요 
        pred = model(x)
        loss = loss_func(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = loss.item()*len(y)
    trn_loss = total_loss/len(dataloader.dataset)
    return trn_loss

# train & test together
def main(cfg):
    from metrics import mape, mae, r2_score
    from tqdm.auto import trange
    from validate import test_one_epoch
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt

    train_params = cfg.get("train_params")
    device = torch.device(train_params.get("device"))

    # get preprocessed csv files
    files = cfg.get("files")
    train_data = pd.read_csv(files.get('train_csv'), index_col=0).to_numpy(dtype=np.float32)
    test_data = pd.read_csv(files.get('test_csv'), index_col=0).to_numpy(dtype=np.float32)

    # Scaler 
    # TODO preprocess로 따로 빼는 것 검토, 
    # TODO config에 scaler 종류 추가하는 것 검토
    # TODO scale아예 하지 않은 경우 고려하는 것 추가 
    scaler = MinMaxScaler()
    trn_scaled = scaler.fit_transform(train_data)
    tst_scaled = scaler.transform(test_data)   ###### TODO train에서 window만큼 데이터 가져와줘야 함.
    # tester에선 csv파일에서 윈도우 사이즈를 고려한 test data를 생성해서 일단 진행

    # Dataset load
    # train
    ds_params = train_params.get("dataset_params")
    dl_params = train_params.get("data_loader_params")
    trn_ds = TimeseriesDataset(trn_scaled, **ds_params)
    trn_dl = DataLoader(trn_ds, **dl_params)
    # x,y = next(iter(trn_dl))
    # print(x.shape, y.shape)
    # print(x.flatten(1).shape, y[:,:,0].shape)

    # test dataset, dataloader
    tst_ds = TimeseriesDataset(tst_scaled, **ds_params)
    tst_dl = DataLoader(tst_ds, shuffle=False, batch_size=len(tst_ds))
    # x,y = next(iter(tst_dl))
    # print(x.shape, y.shape)
    # print(x.flatten(1).shape, y[:,:,0].shape)


    # Model
    Model = cfg.get("model")
    model_params = cfg.get("model_params")
    model = Model(**model_params).to(device)
    # print(model) # To check the model used here
    
    Optim = train_params.get("optim")
    optim_params = train_params.get("optim_params")
    optimizer = Optim(model.parameters(), **optim_params)
    
    loss = train_params.get("loss") # Loss function 
    metric = train_params.get("metric")
    trn_loss_lst = []
    tst_loss_lst = []

    # training and progress bar display
    epochs = train_params.get("epochs")
    pbar = trange(epochs)
    for i in pbar: # loop
        trn_loss = train_one_epoch(trn_dl, model, loss, optimizer, device)
        tst_loss, tst_metric, pred = test_one_epoch(tst_dl, model, loss, metric, device)
        trn_loss_lst.append(trn_loss)
        tst_loss_lst.append(tst_loss)
        pbar.set_postfix({'trn_mse': trn_loss, 'tst_mse': tst_loss, 'tst_mae': tst_metric})
    
    # Save pre-trained weight  -> 재현 용 (저장은 해두기)
    torch.save(model.state_dict(), files.get("model_output"))

    # Prediction
    x, y = next(iter(tst_dl)) # to retrieve x, y from tst_dl
    y = y[:,:,0].to(device)  
    y = y/scaler.scale_[0] + scaler.min_[0] # invejre scaling
    p = pred/scaler.scale_[0] + scaler.min_[0] # inverse scaling
    y = np.concatenate([y[:,0], y[-1,1:]]) # true 
    p = np.concatenate([p[:,0], p[-1,1:]]) # prediction

    print(p) 
    pred_df = pd.DataFrame({'prediction': p})
    pred_df.to_csv(files.get("pred_output"))
    # TODO submission 파일에 pred 넣기

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
    years = list(range(1989, 2009))   # test set에서 불러와서 넣어줘야 함. 
    test_length = 20
    ax2.plot(years[:test_length], y, label = "True")
    ax2.plot(years[:test_length], p, label="Prediction")
    ax2.set_title(f"{Model}, MAPE:{mape(p,y):.4f}, MAE:{mae(p,y):.4f}, R2:{r2_score(p,y):.4f}")
    ax2.legend()
    plt.xticks(years[:test_length:3])
    
    plt.tight_layout()
    plt.savefig(files.get("result_image")) # save as png

### parser
def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Pytorch Model Trainer", add_help=add_help)
    parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    exec(open(args.config).read())
    main(config)