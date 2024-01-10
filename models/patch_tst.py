import torch
from torch import nn
import torch.nn.functional as F

# 각 패치에 있는 특성을 학습하여 시간적인 패턴을 파악
class PatchTST(nn.Module):
  def __init__(self, n_token, input_dim, model_dim, num_heads, num_layers, dim_feedforward, output_dim, output_func):
    super(PatchTST, self).__init__()
    self.patch_embedding = nn.Linear(input_dim, model_dim)    # input_dim = patch_length(16 논문 수치), model_dim = 128(hidden)
    # self.patch_embedding = nn.Linear(input_dim*채널수, model_dim) 
    self._pos = torch.nn.Parameter(torch.randn(1,1,model_dim))  # Positional Embedding
    encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True, dim_feedforward=dim_feedforward) # 128개로 늘려서 어텐션
                                                    # nhead = 16(patch_length와는 관계 없음, 논문 수치)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)  # num_layers = 3 (논문 수치)
    self.output_layer = nn.Linear(model_dim * n_token, output_dim)
    self.output_func = output_func

  def forward(self, x):
    # x shape: (batch_size, n_token, token_size)
    x = self.patch_embedding(x)   # (batch_size, n_token, model_dim) (32, 64, 128)
    x = x + self._pos
    x = self.transformer_encoder(x)   # (batch_size, n_token, model_dim) (32, 64, 128)
    x = x.view(x.size(0), -1) 
    # print(x.view(x.size(0), -1).shape)   # (32, 8192)    # (batch_size, n_token * model_dim) (32, 64*128 =8192)
    output = self.output_layer(x)   # (batch_size, out_dim =24 patch_size == 4)
    return self.output_func(output) # minmax Scaler전용 (scaling 안하면 x나 F.tanh)