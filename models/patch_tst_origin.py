from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from .patch_tst_origin.PatchTST_backbone import PatchTST_backbone
from .patch_tst_origin.PatchTST_layers import series_decomp


class PatchTSTOrigin(nn.Module):
  def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
    act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
    pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
    
    super().__init__()
    
    """
    configs = {
      "enc_in": number_of_features,
      "seq_len": window_size,
      "pred_len": prediction_size,
      "e_layers": n_layers,
      "n_heads": n_heads,
      "d_model": model_dim,
      "d_ff": dimension_of_fully_connected_network,
      "dropout": dropout_ratio,
      "fc_dropout": drop_out_ratio_for_fully_connected_network,
      "head_dropout": head_drop_out,
      "individual": individual_head,
      "patch_len": patch_len,
      "stride": stride,
      "padding_patch": padding_patch,
      "revin": reversible_instance_normalization,
      "affine": ReVIN-affine,
      "subtract_last": subtract_last,
      "decomposition": decomposition,
      "kernel_size": kernel_size
    }
    """
    
    # load parameters
    c_in = configs.enc_in # encoder input size; number of features
    context_window = configs.seq_len # input sequence length; window size??
    target_window = configs.pred_len # prediction sequence length
    
    n_layers = configs.e_layers # number of encoder layers
    n_heads = configs.n_heads # number of heads
    d_model = configs.d_model # model dimension
    d_ff = configs.d_ff # dimension of fully connected network; default=2048
    dropout = configs.dropout # drop out; default=0.05
    fc_dropout = configs.fc_dropout # drop out ratio for fully connected network; default=0.05
    head_dropout = configs.head_dropout # head drop out; default=0.0
    
    individual = configs.individual # individual head; 1 or 0; default=0

    patch_len = configs.patch_len # patch length; default=16
    stride = configs.stride # stride; default=8
    padding_patch = configs.padding_patch # padding; default='end'
    
    revin = configs.revin # RevIn 1 or 0; default=1; Reversible Instance normalization
    affine = configs.affine # RevIn-affine 1 or 0; default=0
    subtract_last = configs.subtract_last # 0: subtract mean; 1: subtract last; default=0
    
    decomposition = configs.decomposition # decomposition; default=0
    kernel_size = configs.kernel_size # kernel size; default=25
    
    
    # model
    self.decomposition = decomposition
    if self.decomposition:
      self.decomp_module = series_decomp(kernel_size)
      self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
        max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
        n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
        dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
        attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
        pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
        pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
        subtract_last=subtract_last, verbose=verbose, **kwargs)
      self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
        max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
        n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
        dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
        attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
        pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
        pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
        subtract_last=subtract_last, verbose=verbose, **kwargs)
    else:
      self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
        max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
        n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
        dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
        attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
        pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
        pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
        subtract_last=subtract_last, verbose=verbose, **kwargs)


def forward(self, x):           # x: [Batch, Input length, Channel]
    if self.decomposition:
      res_init, trend_init = self.decomp_module(x)
      res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
      res = self.model_res(res_init)
      trend = self.model_trend(trend_init)
      x = res + trend
      x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
    else:
      x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
      x = self.model(x)
      x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
    return x