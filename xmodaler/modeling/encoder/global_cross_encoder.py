# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from .build import ENCODER_REGISTRY
from ..layers.positionwise_feedforward import PositionWiseFeedForward
from ..layers.multihead_attention import MultiHeadAttention

__all__ = ["GlobalCrossEncoder"]

class EncoderLayer(nn.Module):
    def __init__(
        self, 
        *,
        d_model=512,  
        num_head=8, 
        num_memory=40,
        d_ff=2048, 
        dropout=.1
    ):
        super(EncoderLayer, self).__init__()
        
        d_k = d_v = d_model // num_head

        self.self_grid = MultiHeadAttention(d_model=d_model,d_k=d_k, d_v=d_v, num_head=num_head, dropout=dropout)
        self.self_region = MultiHeadAttention(d_model=d_model,d_k=d_k, d_v=d_v, num_head=num_head, dropout=dropout)

        self.global_grid = MultiHeadAttention(d_model=d_model,d_k=d_k, d_v=d_v, num_head=num_head, dropout=dropout, shortcut=False)
        self.global_region = MultiHeadAttention(d_model=d_model,d_k=d_k, d_v=d_v, num_head=num_head, dropout=dropout, shortcut=False)

        self.cls_grid = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)
        self.cls_region = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)
        
        self.pwff_grid = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.pwff_region = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, gird_features, region_features, attention_mask):
        b_s = region_features.shape[0]
        cls_grid = self.cls_grid.expand(b_s, 1, -1)
        cls_region = self.cls_region.expand(b_s, 1, -1)
        
        cls_grid = self.global_grid(cls_grid, gird_features, gird_features)
        cls_region = self.global_region(cls_region, region_features, region_features, attention_mask=attention_mask)
        
        gird_features = torch.cat([cls_region, gird_features], dim=1)
        region_features = torch.cat([cls_grid, region_features], dim=1)

        add_mask = torch.zeros(b_s, 1, 1, 1).bool().to(region_features.device)
        attention_mask = torch.cat([add_mask, attention_mask], dim=-1)
        grid_att = self.self_grid(gird_features, gird_features, gird_features)
        region_att = self.self_region(region_features, region_features, region_features, attention_mask=attention_mask)

        gird_ff = self.pwff_grid(grid_att)
        region_ff = self.pwff_region(region_att)

        gird_ff = gird_ff[:,1:]
        region_ff = region_ff[:,1:]
        return gird_ff, region_ff

@ENCODER_REGISTRY.register()
class GlobalCrossEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        input_dim: int, # out_dim of visual embedding
        d_model: int,   # hidden size
        num_layer: int, 
        num_att_head: int, 
        d_ff: int, # feedforward size
        dropout: float,
    ):
        super(GlobalCrossEncoder, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layer
        self.num_att_head = num_att_head
        self.d_ff = d_ff
        self.dropout = dropout

        # encoder hidden layers
        self.layers = nn.ModuleList([EncoderLayer(  d_model=self.d_model,  
                                                    num_head=self.num_att_head, 
                                                    d_ff=self.d_ff, 
                                                    dropout=dropout    
                                                    )
                                     for _ in range(self.num_layers)])

    @classmethod
    def from_config(cls, cfg):

        return {
            "input_dim": cfg.MODEL.VISUAL_EMBED.OUT_DIM,
            "d_model": cfg.MODEL.DFT.ENCODER.DIM_MODEL,
            "num_layer": cfg.MODEL.DFT.ENCODER.NUM_LAYER,
            "num_att_head": cfg.MODEL.DFT.ENCODER.NUM_ATT_HEAD,
            "d_ff": cfg.MODEL.DFT.ENCODER.DIM_FEEDFORWARD,
            "dropout": cfg.MODEL.DFT.ENCODER.DROPOUT,
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.DFT = CN()
        
        cfg.MODEL.DFT.ENCODER = CN()

        cfg.MODEL.DFT.ENCODER.DIM_MODEL = 512
        cfg.MODEL.DFT.ENCODER.NUM_LAYER = 3
        cfg.MODEL.DFT.ENCODER.DROPOUT = 0.1
        cfg.MODEL.DFT.ENCODER.NUM_ATT_HEAD = 8
        cfg.MODEL.DFT.ENCODER.DIM_FEEDFORWARD = 2048

    def forward(self, batched_inputs, mode=None):
        ret = {}
        if mode == None or mode == 'v':
            region_feats = batched_inputs[kfg.ATT_FEATS]
            region_masks = batched_inputs[kfg.ATT_MASKS]
            region_attention_mask = (region_masks == 0)
            grid_feats = batched_inputs[kfg.ATT_FEATS_WO_MASK]

            for l in self.layers:
                grid_feats, region_feats = l(grid_feats, region_feats, region_attention_mask)

            ret.update({
                kfg.ATT_FEATS: region_feats,
                kfg.ATT_MASKS: region_attention_mask,
                kfg.ATT_FEATS_WO_MASK: grid_feats
            })
        return ret