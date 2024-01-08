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
        x, y = x.flatten(1).to(device), y[:,0].unsqueeze(1).to(device) # 2D로 맞춰줌
        # TODO enefit에서는 y[] index 맞춰줄때 어떤 column을 타겟으로 하는지 확인 필요 
        with torch.inference_mode():
            pred = model(x)
            loss = loss_func(pred, y)
            metric_value = metric(pred, y)
            total_loss = loss.item()*len(y)
            total_metric = metric_value.item()*len(y)
    tst_loss = total_loss/len(tst_dl.dataset)
    tst_metric = total_metric/len(tst_dl.dataset)
    pred = pred.cpu().numpy()
    return tst_loss, tst_metric, pred  
