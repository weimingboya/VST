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

__all__ = ["MMADecoder"]

class BertSelfOutput(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        hidden_size: int,
        layer_norm_eps: float,
        hidden_dropout_prob: float
    ):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size * 3, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "layer_norm_eps": 1e-12,
            "hidden_dropout_prob": cfg.MODEL.BERT.HIDDEN_DROPOUT_PROB
        }

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertGenerationLayer(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        bert_attention,
        # bert_comm_cross_attention,
        bert_grid_cross_attention,
        bert_text_cross_attention,
        bert_self_output,
        bert_mix_cross_attention,
        bert_intermediate,
        bert_output,

    ):
        super(BertGenerationLayer, self).__init__()
        self.self_attn = bert_attention
        # self.comm_cross_att = bert_comm_cross_attention
        self.grid_cross_att = bert_grid_cross_attention
        self.text_cross_att = bert_text_cross_attention
        self.cross_output = bert_self_output
        self.mix_cross_attention = bert_mix_cross_attention
        self.intermediate = bert_intermediate
        self.output = bert_output
        # self.v2t_cross_atts = []
        # self.t2v_cross_atts = []
        # self.vt_cross_atts = []

    @classmethod
    def from_config(cls, cfg):
        return {
            "bert_attention": BertAttention(cfg),
            # "bert_comm_cross_attention": BertCrossAttention(cfg),
            "bert_grid_cross_attention": BertCrossAttention(cfg),
            "bert_text_cross_attention": BertCrossAttention(cfg),
            "bert_self_output": BertSelfOutput(cfg),
            "bert_mix_cross_attention": BertCrossAttention(cfg),
            "bert_intermediate": BertIntermediate(cfg),
            "bert_output": BertOutput(cfg),
        }

    def forward(self, lang_feats, visn_feats, text_feats, cat_feats, lang_attention_mask=None, v_attention_mask=None, mix_attention_mask=None, t_history_states=None):
        lang_feats, _ = self.self_attn(lang_feats, lang_attention_mask, t_history_states)
        visn_feats, v_cross_att = self.grid_cross_att(lang_feats, visn_feats, visn_feats, v_attention_mask, None)
        text_feats, t_cross_att = self.text_cross_att(lang_feats, text_feats, text_feats, None, None)
        cat_feats, vt_cross_att = self.mix_cross_attention(lang_feats, cat_feats, cat_feats, mix_attention_mask, None)
        
        # self.v2t_cross_atts.append(v_cross_att)
        # self.t2v_cross_atts.append(t_cross_att)
        # self.vt_cross_atts.append(vt_cross_att)
        
        final_feats = torch.cat([visn_feats, text_feats, cat_feats], dim=-1)
        final_feats = self.cross_output(final_feats, lang_feats)
        
        intermediate_output = self.intermediate(final_feats)
        final_feats = self.output(intermediate_output, final_feats)

        return final_feats

@DECODER_REGISTRY.register()
class MMADecoder(Decoder):
    @configurable
    def __init__(
        self,
        *,
       num_generation_layers: int,
       bert_generation_layers
    ):
        super(MMADecoder, self).__init__()
        self.num_generation_layers = num_generation_layers
        if self.num_generation_layers > 0:
            self.g_layers = bert_generation_layers

    @classmethod
    def from_config(cls, cfg):
        bert_generation_layers = nn.ModuleList(
            [BertGenerationLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_GENERATION_LAYERS)]
        )
        return {
            "num_generation_layers": cfg.MODEL.BERT.NUM_GENERATION_LAYERS,
            "bert_generation_layers": bert_generation_layers,
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        ret = {}
        visn_feats = batched_inputs[kfg.ATT_FEATS]
        ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS]
        text_feats = batched_inputs[kfg.ATT_FEATS_WO_MASK]
        history_states = batched_inputs.get(kfg.HISTORY_STATES, None)
        cat_feats = torch.cat([visn_feats, text_feats], dim=1)
        text_vmasks = torch.zeros((text_feats.shape[0], 1, 1, text_feats.shape[1]), device=text_feats.device)
        cat_ext_vmasks = torch.cat([ext_vmasks, text_vmasks], dim=-1)
        
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

        for i, layer_module in enumerate(self.g_layers):
            if history_states[i] is not None:
                history_states[i] = torch.cat([history_states[i], g_tfeats], dim=1)

            g_tfeats = layer_module(g_tfeats, visn_feats, text_feats, cat_feats, ext_g_tmasks, ext_vmasks, cat_ext_vmasks, history_states[i])
            g_tfeats_arr.append(g_tfeats)
        ret.update({ kfg.G_HIDDEN_STATES: g_tfeats_arr })

        return ret