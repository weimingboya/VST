# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import math
import numpy as np
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from xmodaler.modeling.layers.bert import BertAttention, BertCrossAttention, BertIntermediate, BertOutput, BertSelfOutput
from .build import ENCODER_REGISTRY

__all__ = ["TransformeVideoEncoder"]

class BertLayer(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        bert_attention, 
        bert_cross_attention, 
        bert_intermediate,
        bert_output
    ):
        super(BertLayer, self).__init__()
        self.attention = bert_attention
        self.bert_cross_attention = bert_cross_attention
        self.intermediate = bert_intermediate
        self.output = bert_output

    @classmethod
    def from_config(cls, cfg):
        return {
            "bert_attention": BertAttention(cfg),
            "bert_cross_attention": BertCrossAttention(cfg),
            "bert_intermediate": BertIntermediate(cfg),
            "bert_output": BertOutput(cfg)
        }

    def forward(self, hidden_states, attention_mask, concept_protos, history_states=None):
    # def forward(self, hidden_states, attention_mask, history_states=None):
        attention_output, attention_probs = self.attention(hidden_states, attention_mask, history_states)
        attention_output,_ = self.bert_cross_attention(attention_output, concept_protos, concept_protos, None, None)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs

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
        self.embed = nn.Linear(768, 512)

        # self.concept_protos = []
        # self.concept_protos.append(torch.load('datasets/msvd_dataset/hyper_protos.pth')['hyper2k-800'])
        # self.concept_protos.append(torch.load('datasets/msvd_dataset/hyper_protos.pth')['hyper2k'])
        self.concept_protos = torch.load('datasets/msvd_dataset/hyper_protos.pth')['hyper2k']

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

            for i, layer_module in enumerate(self.layers):
                concepts = self.concept_protos.to(global_feats.device).unsqueeze(0).repeat(global_feats.shape[0], 1, 1)
                concepts = self.embed(concepts)
                global_feats, _ = layer_module(global_feats, ext_vmasks, concepts)
                # global_feats, _ = layer_module(global_feats, ext_vmasks)

            ret.update( {kfg.ATT_FEATS_WO_MASK: grid_feats, kfg.ATT_FEATS: global_feats })
            # ret.update( {kfg.ATT_FEATS: global_feats })
        return ret