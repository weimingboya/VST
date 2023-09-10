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
from ..layers.multihead_attention import MultiHeadAttention, TopDownAttention
from ..layers.positionwise_feedforward import PositionWiseFeedForward

from .decoder import Decoder
from .build import DECODER_REGISTRY

__all__ = ["VSDecoder"]

class DecoderLayer(nn.Module):
    def __init__(
        self, 
        *,
        d_model=512, 
        num_head=8, 
        d_ff=2048, 
        dropout=.1,
    ):
        super(DecoderLayer, self).__init__()

        d_k = d_v = d_model // num_head

        self.self_att = MultiHeadAttention(d_model=d_model,d_k=d_k, d_v=d_v, num_head=num_head, dropout=dropout)
        self.text_att = MultiHeadAttention(d_model=d_model,d_k=d_k, d_v=d_v, num_head=num_head, dropout=dropout, shortcut=False)
        self.region_att = MultiHeadAttention(d_model=d_model,d_k=d_k, d_v=d_v, num_head=num_head, dropout=dropout, shortcut=False)
        self.pwff = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.gate1 = nn.Sequential(
            nn.Linear(2 * d_model, 1),
            nn.Sigmoid()
        )

        self.gate2 = nn.Sequential(
            nn.Linear(2 * d_model, 1),
            nn.Sigmoid()
        )

        self.ln = nn.LayerNorm(d_model)

    def forward(self, input, region_feats, text_feats , mask_self_att, mask_enc_att, history_states=None):
        self_att = self.self_att(input, input, input, mask_self_att, history_states=history_states)
       
        region_att = self.region_att(self_att, region_feats, region_feats, mask_enc_att)
        text_att = self.text_att(self_att, text_feats, text_feats)

        alpha1 = self.gate1(torch.cat([region_att, self_att], -1))
        alpha2 = self.gate2(torch.cat([text_att, self_att], -1))

        enc_att = self.ln(alpha1 * region_att + alpha2 * text_att + self_att)
        
        ff = self.pwff(enc_att)
        return ff

@DECODER_REGISTRY.register()
class VSDecoder(Decoder):
    @configurable
    def __init__(
        self,
        *,
        d_model: int , 
        num_layer: int,  
        num_att_head: int, 
        d_ff: int, 
        dropout: float,
        padding_idx: int, # -1
    ):
        super(VSDecoder, self).__init__()

        self.num_layers = num_layer
        self.d_model = d_model
        self.num_att_head = num_att_head
        self.d_ff = d_ff
        self.dropout = dropout
        self.padding_idx = padding_idx

        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=self.d_model, 
                num_head=self.num_att_head, 
                d_ff=self.d_ff, 
                dropout=self.dropout,
            ) for _ in range(self.num_layers)
        ])

    @classmethod
    def from_config(cls, cfg):
        return {
            "d_model": cfg.MODEL.VST.DECODER.DIM_MODEL,
            "num_layer": cfg.MODEL.VST.DECODER.NUM_LAYER,
            "num_att_head": cfg.MODEL.VST.DECODER.NUM_ATT_HEAD,
            "d_ff": cfg.MODEL.VST.DECODER.DIM_FEEDFORWARD,
            "dropout": cfg.MODEL.VST.DECODER.DROPOUT,
            "padding_idx": -1, # default
        }

    @classmethod
    def add_config(cls, cfg):
        if not hasattr(cfg.MODEL, "VST"):
            cfg.MODEL.VST = CN()

        cfg.MODEL.VST.DECODER = CN()
        cfg.MODEL.VST.DECODER.DIM_MODEL = 512
        cfg.MODEL.VST.DECODER.NUM_LAYER = 3
        cfg.MODEL.VST.DECODER.DROPOUT = 0.1
        cfg.MODEL.VST.DECODER.NUM_ATT_HEAD = 8
        cfg.MODEL.VST.DECODER.DIM_FEEDFORWARD = 2048

    def forward(self, batched_inputs):
        ret = {}
        region_feats = batched_inputs[kfg.ATT_FEATS]
        region_masks = batched_inputs[kfg.ATT_MASKS]
        text_feats = batched_inputs[kfg.ATT_FEATS_WO_MASK]

        history_states = batched_inputs.get(kfg.HISTORY_STATES, None)

        g_tfeats_arr = []
        g_tfeats = batched_inputs[kfg.G_TOKEN_EMBED]
        ext_g_tmasks = batched_inputs[kfg.EXT_G_TOKENS_MASKS]
        ext_g_tmasks = (ext_g_tmasks == -10000.0) # FIXME
        if len(g_tfeats.size()) == 2:
            g_tfeats = g_tfeats.unsqueeze(1)

        if kfg.TIME_STEP in batched_inputs:
            time_step = batched_inputs[kfg.TIME_STEP]
            ext_g_tmasks = ext_g_tmasks[:,:, time_step:time_step+1, 0:time_step+1]
            if kfg.HISTORY_STATES not in batched_inputs:
                shape = list(g_tfeats.size())
                shape[1] = 0
                history_states = [g_tfeats.new(torch.Size(shape))] * self.num_layers
                batched_inputs[kfg.HISTORY_STATES] = history_states
        else:
            history_states = [None] * self.num_layers

        for i, layer_module in enumerate(self.layers):
            if history_states[i] is not None:
                history_states[i] = torch.cat([history_states[i], g_tfeats], dim=1)

            g_tfeats = layer_module(g_tfeats, region_feats, text_feats, ext_g_tmasks, region_masks, history_states[i])
            g_tfeats_arr.append(g_tfeats)
        ret.update({ kfg.G_HIDDEN_STATES: g_tfeats_arr })

        return ret