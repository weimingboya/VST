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
# from ..layers.bert import BertAttention, BertCrossAttention, BertIntermediate, BertOutput
from ..layers.bert_pe import BertAttention, BertCrossAttention, BertIntermediate, BertOutput
from .build import DECODER_REGISTRY

__all__ = ["TransformerCustomDecoder"]

class BertGenerationLayer(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        bert_attention,
        bert_visual_cross_attention,
        # bert_text_cross_attention,
        bert_intermediate,
        bert_output
    ):
        super(BertGenerationLayer, self).__init__()
        self.self_attn = bert_attention
        self.visual_att = bert_visual_cross_attention
        # self.text_att = bert_text_cross_attention
        self.intermediate = bert_intermediate
        self.output = bert_output

        # self.project = nn.Sequential(
        #     nn.Linear(512*2,512),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )

        self.fc_alpha1 = nn.Linear(512*2, 512)
        self.fc_alpha2 = nn.Linear(512*2, 512)
        # self.layernorm = nn.LayerNorm(512)

        # d_model = 512
        # self.aoa1 = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model), nn.GLU())
        # self.aoa2 = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model), nn.GLU())

    @classmethod
    def from_config(cls, cfg):
        return {
            "bert_attention": BertAttention(cfg),
            "bert_visual_cross_attention": BertCrossAttention(cfg),
            # "bert_text_cross_attention": BertCrossAttention(cfg),
            "bert_intermediate": BertIntermediate(cfg),
            "bert_output": BertOutput(cfg)
        }

    def forward(self, lang_feats, visn_feats, text_feats, region_ape, text_ape, lang_attention_mask=None, v_attention_mask=None, t_history_states=None):
        lang_feats, _ = self.self_attn(lang_feats, lang_attention_mask, t_history_states)
        visn_feats, _ = self.visual_att(lang_feats, visn_feats, visn_feats, v_attention_mask, None, None, region_ape)
        # text_feats, _ = self.text_att(lang_feats, text_feats, text_feats, None, None, None, text_ape)
        text_feats, _ = self.visual_att(lang_feats, text_feats, text_feats, None, None, None, text_ape)
        
        # x = visn_feats + text_feats
        # x = self.project(torch.cat([visn_feats, text_feats], dim=-1))

        alpha1 = self.fc_alpha1(torch.cat([lang_feats, visn_feats], -1))
        alpha2 = self.fc_alpha2(torch.cat([lang_feats, text_feats], -1))

        alpha1 = torch.sigmoid(alpha1)
        alpha2 = torch.sigmoid(alpha2)

        x = (visn_feats * alpha1 + text_feats * alpha2) / np.sqrt(2)
        # x = self.layernorm(visn_feats * alpha1 + text_feats * alpha2)

        # visn_feats = self.aoa1(torch.cat([visn_feats, lang_feats], -1))
        # text_feats = self.aoa2(torch.cat([text_feats, lang_feats], -1))

        # x = grid_feats + text_feats
        
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
       bert_generation_layers
    ):
        super(TransformerCustomDecoder, self).__init__()
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

        pos_emb = batched_inputs[kfg.POS_EMBED]
        n_text = 180
        text_ape, region_ape = pos_emb[:,:n_text], pos_emb[:,n_text:]

        for i, layer_module in enumerate(self.g_layers):
            if history_states[i] is not None:
                history_states[i] = torch.cat([history_states[i], g_tfeats], dim=1)

            g_tfeats = layer_module(g_tfeats, visn_feats, text_feats, region_ape, text_ape, ext_g_tmasks, ext_vmasks, history_states[i])
            # g_tfeats = layer_module(g_tfeats, visn_feats, text_feats, None, None, ext_g_tmasks, ext_vmasks, history_states[i])
            g_tfeats_arr.append(g_tfeats)
        ret.update({ kfg.G_HIDDEN_STATES: g_tfeats_arr })

        return ret