import torch
import torch.nn as nn
import torch.nn.functional as F

class ANNMulti(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, input_channel, activation = F.relu):
        super().__init__()
        self.lin1 = nn.Linear(input_dim * input_channel, hidden_dim) # flatten 해서 들어가는 것처럼, 곱하기 해주어야 하는건가?
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        self.activation = activation
    def forward(self, x):
        x = x.flatten(1) # (Batch, input_dim, input_channel) -> (Batch, input_dim * input_channel)
        x = self.lin1(x) # (Batch, hidden_dim)
        x = self.activation(x)
        x = self.lin2(x) # (Batch, output_dim)
        return F.sigmoid(x)