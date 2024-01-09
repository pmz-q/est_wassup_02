from tqdm.auto import trange
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Optimizer
from .nn_validate import test_one_epoch



# train function for one epoch
def train_one_epoch(dataloader, model, loss_func, optimizer, device, scheduler):
  model.train()
  total_loss = 0. 
  for x, y in dataloader: 
    x, y = x.flatten(1).to(device), y[:,0].unsqueeze(1).to(device) # 2D로 맞춰줌 -> 
    # TODO enefit에서는 y[] index 맞춰줄때 어떤 column을 타겟으로 하는지 확인 필요 
    # y[:,:,0] target_column_index를 받아서 0대신 변수로 넣어줄 수 있음. 
    pred = model(x)
    # print(pred)
    # print(y)
    loss = loss_func(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item()*len(y)
  trn_loss = total_loss/len(dataloader.dataset)
  if scheduler != None: scheduler.step()
  return trn_loss

def nn_train(
  epochs:int, trn_dl: DataLoader, tst_dl: DataLoader, model: nn.Module,
  loss: callable, metric: callable, optimizer: Optimizer, device, 
  scheduler, use_early_stop: bool, early_stop_params: dict
):
  best_loss = 10 ** 9
  patience_limit = early_stop_params.get("patience_limit")
  patience_check = 0
  
  trn_loss_lst = []
  tst_loss_lst = []

  # training and progress bar display
  pbar = trange(epochs)
  for i in pbar: # loop
    trn_loss = train_one_epoch(trn_dl, model, loss, optimizer, device, scheduler)
    tst_loss, tst_metric, pred = test_one_epoch(tst_dl, model, loss, metric, device)
    trn_loss_lst.append(trn_loss)
    tst_loss_lst.append(tst_loss)
    pbar.set_postfix({'trn_mse': trn_loss, 'tst_mse': tst_loss, 'tst_mae': tst_metric})

    # EARLY STOP
    if use_early_stop and tst_loss > best_loss:
      patience_check += 1
      if patience_check >= patience_limit: break
    else:
      best_loss = tst_loss
      patience_check = 0
  
  return trn_loss_lst, tst_loss_lst, pred
  