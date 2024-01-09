from tqdm.auto import trange
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Optimizer
from .tst_validate import test_one_epoch



# train function for one epoch
def train_one_epoch(dataloader, model, loss_func, optimizer, device):
  model.train()
  total_loss = 0. 
  for x, y in dataloader: 
    x, y = x.to(device), y.to(device) 
    pred = model(x)
    # print(pred)
    # print(y)
    loss = loss_func(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item()*len(y)
  trn_loss = total_loss/len(dataloader.dataset)
  return trn_loss

def tst_train(
  epochs:int, trn_dl: DataLoader, tst_dl: DataLoader, model: nn.Module,
  loss: callable, metric: callable, optimizer: Optimizer, device 
):
  trn_loss_lst = []
  tst_loss_lst = []

  # training and progress bar display
  pbar = trange(epochs)
  for i in pbar: # loop
    trn_loss = train_one_epoch(trn_dl, model, loss, optimizer, device)
    tst_loss, tst_metric, pred = test_one_epoch(tst_dl, model, loss, metric, device)
    trn_loss_lst.append(trn_loss)
    tst_loss_lst.append(tst_loss)
    pbar.set_postfix({'trn_mse': trn_loss, 'tst_mse': tst_loss, 'tst_mae': tst_metric})
  
  return trn_loss_lst, tst_loss_lst, pred
  