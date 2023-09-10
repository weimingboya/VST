# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import numpy as np
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from .decoder import Decoder
from ..layers.bert import BertAttention, BertCrossAttention, BertIntermediate, BertOutput, BertXAttention
from .build import DECODER_REGISTRY

__all__ = ["TransformerCustomDecoder"]

# class AttentionPool(nn.Module):
#     @configurable
#     def __init__(
#         self, 
#         bert_self_attention,
#         bert_self_output
#     ):
#         super(BertAttention, self).__init__()
#         self.self = bert_self_attention
#         self.output = bert_self_output
        
#     @classmethod
#     def from_config(cls, cfg):
#         return {
#             "bert_attention": BertAttention(cfg),
#             "bert_cross_attention": BertCrossAttention(cfg),
#             "bert_intermediate": BertIntermediate(cfg),
#             "bert_output": BertOutput(cfg)
#         }

class BertGenerationLayer(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        bert_attention,
        bert_cross_attention,
        bert_intermediate,
        bert_output
    ):
        super(BertGenerationLayer, self).__init__()
        self.self_attn = bert_attention
        self.cross_att = bert_cross_attention
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

    def forward(self, lang_feats, mix_feats, lang_attention_mask=None, v_attention_mask=None, t_history_states=None):
        lang_feats, _ = self.self_attn(lang_feats, lang_attention_mask, t_history_states)
        x, _ = self.cross_att(lang_feats, mix_feats, mix_feats, v_attention_mask, None)

        intermediate_output = self.intermediate(x)
        layer_output = self.output(intermediate_output, x)

        return layer_output

@DECODER_REGISTRY.register()
class TransformerCustomDecoder(Decoder):
    @configurable
    def __init__(
        self,
        *,
       num_generation_layers: int,
       bert_generation_layers,
    #    bert_cross_attention,
    #    bert_intermediate,
    #    bert_output
    ):
        super(TransformerCustomDecoder, self).__init__()
        self.num_generation_layers = num_generation_layers
        if self.num_generation_layers > 0:
            self.g_layers = bert_generation_layers
            
        # self.queries = nn.Parameter(torch.randn(128, 512), requires_grad=True)
        # self.bert_cross_attention = bert_cross_attention
        # self.intermediate = bert_intermediate
        # self.output = bert_output

    @classmethod
    def from_config(cls, cfg):
        bert_generation_layers = nn.ModuleList(
            [BertGenerationLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_GENERATION_LAYERS)]
        )
        return {
            "num_generation_layers": cfg.MODEL.BERT.NUM_GENERATION_LAYERS,
            "bert_generation_layers": bert_generation_layers,
            # "bert_cross_attention": BertXAttention(cfg),
            # "bert_intermediate": BertIntermediate(cfg),
            # "bert_output": BertOutput(cfg)
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        ret = {}
        mix_feats = batched_inputs[kfg.ATT_FEATS]
        ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS]
        history_states = batched_inputs.get(kfg.HISTORY_STATES, None)

        g_tfeats_arr = []
        g_tfeats = batched_inputs[kfg.G_TOKEN_EMBED]
        ext_g_tmasks = batched_inputs[kfg.EXT_G_TOKENS_MASKS]
        if len(g_tfeats.size()) == 2:
            g_tfeats = g_tfeats.unsqueeze(1)
        
        if kfg.TIME_STEP in batched_inputs:
            time_step = batched_inputs[kfg.TIME_STEP]
            ext_g_tmasks = ext_g_tmasks[:,:, time_step:time_step+1, 0:time_step+1]
            if kfg.HISTORY_STATES not in batched_inputs:
                shape = list(g_tfeats.size())
                shape[1] = 0
                history_states = [g_tfeats.new(torch.Size(shape))] * self.num_generation_layers
                batched_inputs[kfg.HISTORY_STATES] = history_states
        else:
            history_states = [None] * self.num_generation_layers

        # queries = self.queries.unsqueeze(0).expand(mix_feats.shape[0], *self.queries.shape).contiguous()
        # queries, _ = self.bert_cross_attention(queries, mix_feats, mix_feats, ext_vmasks)
        # intermediate_output = self.intermediate(queries)
        # layer_output = self.output(intermediate_output, queries)
        
        for i, layer_module in enumerate(self.g_layers):
            if history_states[i] is not None:
                history_states[i] = torch.cat([history_states[i], g_tfeats], dim=1)

            g_tfeats = layer_module(g_tfeats, mix_feats, ext_g_tmasks, ext_vmasks, history_states[i])
            # g_tfeats = layer_module(g_tfeats, queries, ext_g_tmasks, None, history_states[i])
            g_tfeats_arr.append(g_tfeats)
        ret.update({ kfg.G_HIDDEN_STATES: g_tfeats_arr })

        return ret