# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from xmodaler.modeling.layers.bert import BertLayer
from .build import ENCODER_REGISTRY

__all__ = ["TransformeVideoEncoder"]

@ENCODER_REGISTRY.register()
class TransformerVideoEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        num_hidden_layers: int,
        bert_layers,
    ):
        super(TransformerVideoEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.layers = bert_layers

    @classmethod
    def from_config(cls, cfg):
        bert_layers = nn.ModuleList(
            [BertLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_HIDDEN_LAYERS)]
        )
        return {
            "num_hidden_layers": cfg.MODEL.BERT.NUM_HIDDEN_LAYERS,
            "bert_layers": bert_layers,
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs, mode=None):
        ret = {}
        if mode == None or mode == 'v':
            grid_feats = batched_inputs[kfg.ATT_FEATS_WO_MASK]
            ext_vmasks = None
            global_feats = batched_inputs[kfg.ATT_FEATS]

            # gv = global_feats.mean(dim=1, keepdim=True)
            # global_feats = torch.cat([gv, global_feats], dim=1)  # (bs,n+1,d)

            for layer_module in self.layers:
                global_feats, _ = layer_module(global_feats, ext_vmasks)

            ret.update( {kfg.ATT_FEATS_WO_MASK: grid_feats, kfg.ATT_FEATS: global_feats })
        return ret