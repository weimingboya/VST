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

__all__ = ["VSEncoder"]

class EncoderLayer(nn.Module):
    def __init__(
        self, 
        *,
        d_model=512,  
        num_head=8, 
        d_ff=2048, 
        dropout=.1
    ):
        super(EncoderLayer, self).__init__()
        
        d_k = d_v = d_model // num_head

        self.self_text = MultiHeadAttention(d_model=d_model,d_k=d_k, d_v=d_v, num_head=num_head, dropout=dropout)
        self.self_region = MultiHeadAttention(d_model=d_model,d_k=d_k, d_v=d_v, num_head=num_head, dropout=dropout)

        self.pwff_text = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.pwff_region = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, region_feats, text_feats, attention_mask):
        region_att = self.self_region(region_feats, region_feats, region_feats, attention_mask=attention_mask)
        text_att = self.self_text(text_feats, text_feats, text_feats)

        region_ff = self.pwff_region(region_att)
        text_ff = self.pwff_text(text_att)

        return region_ff, text_ff

@ENCODER_REGISTRY.register()
class VSEncoder(nn.Module):
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
        super(VSEncoder, self).__init__()

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
            "d_model": cfg.MODEL.VST.ENCODER.DIM_MODEL,
            "num_layer": cfg.MODEL.VST.ENCODER.NUM_LAYER,
            "num_att_head": cfg.MODEL.VST.ENCODER.NUM_ATT_HEAD,
            "d_ff": cfg.MODEL.VST.ENCODER.DIM_FEEDFORWARD,
            "dropout": cfg.MODEL.VST.ENCODER.DROPOUT,
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.VST = CN()
        
        cfg.MODEL.VST.ENCODER = CN()

        cfg.MODEL.VST.ENCODER.DIM_MODEL = 512
        cfg.MODEL.VST.ENCODER.NUM_LAYER = 3
        cfg.MODEL.VST.ENCODER.DROPOUT = 0.1
        cfg.MODEL.VST.ENCODER.NUM_ATT_HEAD = 8
        cfg.MODEL.VST.ENCODER.DIM_FEEDFORWARD = 2048

    def forward(self, batched_inputs, mode=None):
        ret = {}
        if mode == None or mode == 'v':
            region_feats = batched_inputs[kfg.ATT_FEATS]
            region_masks = batched_inputs[kfg.ATT_MASKS]
            region_attention_mask = (region_masks == 0)
            text_feats = batched_inputs[kfg.ATT_FEATS_WO_MASK]

            for l in self.layers:
                region_feats, text_feats = l(region_feats, text_feats, region_attention_mask)

            ret.update({
                kfg.ATT_FEATS: region_feats,
                kfg.ATT_MASKS: region_attention_mask,
                kfg.ATT_FEATS_WO_MASK: text_feats
            })
        return ret