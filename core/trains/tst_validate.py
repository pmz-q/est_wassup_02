import torch
from torch import nn
from torch.utils.data import DataLoader

def test_one_epoch(
  tst_dl: DataLoader, model: nn.Module, loss_func: callable,
  metric: callable, device
):
    model.eval()
    total_loss, total_metric = 0., 0.
    for x, y in tst_dl:
        x, y = x.to(device), y.to(device) 
        with torch.inference_mode():
            pred = model(x)
            loss = loss_func(pred, y)
            metric_value = metric(pred, y)
            total_loss += loss.item()*len(y)
            total_metric += metric_value.item()*len(y)
    tst_loss = total_loss/len(tst_dl.dataset)
    tst_metric = total_metric/len(tst_dl.dataset)
    pred = pred.cpu().numpy()
    return tst_loss, tst_metric, pred  
    #TODO pred loop마다 불러오지 않고 저장만했다가 마지막 prediction만 내보내기
