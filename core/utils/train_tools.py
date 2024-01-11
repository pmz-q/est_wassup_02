import torch


def save_best_model(output_path: str) -> callable:
  best_valid_loss = float('inf')
  def save(current_valid_loss, epoch, model, optimizer, criterion):
    if current_valid_loss < best_valid_loss:
      best_valid_loss = current_valid_loss
      print(f"\nBest validation loss: {best_valid_loss}")
      print(f"\nSaving best model for epoch: {epoch+1}\n")
      torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
      }, output_path)
  return save