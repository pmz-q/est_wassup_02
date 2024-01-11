import torch


class SaveBestModel():
  def __init__(self, output_path: str, best_val_loss=float('inf')):
    self.best_val_loss = best_val_loss
    self.output_path = output_path
  
  def __call__(
    self, val_loss, model
  ):
    if val_loss < self.best_val_loss:
      self.best_val_loss = val_loss
      torch.save(model.state_dict(), self.output_path)
