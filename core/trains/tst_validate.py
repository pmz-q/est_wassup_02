import torch
from torch import nn
from torch.utils.data import DataLoader

def validate_one_epoch(
  val_dl: DataLoader, model: nn.Module, loss_func: callable,
  metric: callable, device
):
    model.eval()
    total_loss, total_metric = 0., 0.
    for x, y in val_dl:
        x, y = x.to(device), y.to(device) 
        with torch.inference_mode():
            pred = model(x)
            loss = loss_func(pred, y)
            metric_value = metric(pred, y)
            total_loss += loss.item()*len(y)
            total_metric += metric_value.item()*len(y)
    val_loss = total_loss/len(val_dl.dataset)
    val_metric = total_metric/len(val_dl.dataset)
    return val_loss, val_metric
