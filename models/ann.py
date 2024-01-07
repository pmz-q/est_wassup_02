import torch.nn as nn
import torch.nn.functional as F

class ANN(nn.Module):
  def __init__(
    self,
    input_dim: int=5,
    input_channel: int=2,
    hidden_dim: list=[128, 128, 64, 32],
    activation: callable=F.relu,
    use_drop:bool = True,
    drop_ratio: float=0.0,
    output_dim: int=1
  ):
    super().__init__()
    dims = [input_dim*input_channel] + hidden_dim
    self.identity = nn.Identity()
    self.dropout = nn.Dropout(drop_ratio)
    self.activation = activation
    
    model = [[nn.Linear(dims[i], dims[i+1]), nn.BatchNorm1d(dims[i+1]), self.dropout if use_drop else self.identity, self.activation] for i in range(len(dims) - 1)]
    output_layer = [nn.Linear(dims[-1], output_dim), nn.Identity()]
    self.module_list= nn.ModuleList(sum(model, []) + output_layer)
  
  def forward(self, x):
    for layer in self.module_list:
         x = layer(x)
    return x