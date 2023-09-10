# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import math
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from xmodaler.modeling.embedding.position_embedding import BoxRelationalEmbedding
from xmodaler.modeling.layers.bert_pe import BertAttention, BertCrossAttention
from xmodaler.modeling.layers.create_act import get_activation
from .build import ENCODER_REGISTRY
from torch.nn import functional as F

__all__ = ["GPSCAEncoder"]

class BertFFN(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        hidden_size: int,
        hidden_act: str,
        intermediate_size: int,
        intermediate_drop: float,
        layer_norm_eps: float,
        ffn_dropout_prob: float
    ):
        super(BertFFN, self).__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = get_activation(hidden_act)
        self.dropout1 = nn.Dropout(intermediate_drop)

        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(ffn_dropout_prob)

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "hidden_act": cfg.MODEL.BERT.HIDDEN_ACT,
            "intermediate_size": cfg.MODEL.BERT.INTERMEDIATE_SIZE,
            "intermediate_drop": cfg.MODEL.BERT.INTERMEDIATE_DROP,
            "layer_norm_eps": 1e-12,
            "ffn_dropout_prob": cfg.MODEL.BERT.FFN_DROPOUT_PROB
        }

    def forward(self, hidden_states):
        input_tensor = hidden_states
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout1(hidden_states)

        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        bert_comm_self_attention, 
        bert_visn_self_FFN, 
        bert_text_self_FFN,
        bert_comm_cross_attention,
        bert_visn_cross_FFN,
        bert_text_cross_FFN,
    ):
        super(BertLayer, self).__init__()
        self.bert_comm_self_attention = bert_comm_self_attention
        self.bert_visn_self_FFN = bert_visn_self_FFN
        self.bert_text_self_FFN = bert_text_self_FFN
        self.bert_comm_cross_attention = bert_comm_cross_attention
        self.bert_visn_cross_FFN = bert_visn_cross_FFN
        self.bert_text_cross_FFN = bert_text_cross_FFN

    @classmethod
    def from_config(cls, cfg):
        return {
            "bert_comm_self_attention": BertAttention(cfg),
            "bert_visn_self_FFN": BertFFN(cfg),
            "bert_text_self_FFN": BertFFN(cfg),
            
            "bert_comm_cross_attention": BertCrossAttention(cfg),
            "bert_visn_cross_FFN": BertFFN(cfg),
            "bert_text_cross_FFN": BertFFN(cfg),
        }

    def forward(self, visn_feats, text_feats, ext_vmasks, region_ape, text_ape, r2r_rpe, r2t_rpe, t2t_rpe, t2r_rpe):
        visn_seff_att, _ = self.bert_comm_self_attention(visn_feats, ext_vmasks, None, region_ape, r2r_rpe)
        text_seff_att, _ = self.bert_comm_self_attention(text_feats, None, None, text_ape, t2t_rpe)
        visn_seff_ff = self.bert_visn_self_FFN(visn_seff_att)
        text_seff_ff = self.bert_text_self_FFN(text_seff_att)

        visn_cross_att, v2t_att_map = self.bert_comm_cross_attention(visn_seff_ff, text_seff_ff, text_seff_ff, None, None, 
                                                           region_ape, text_ape, r2t_rpe)
        text_cross_att, t2v_att_map = self.bert_comm_cross_attention(text_seff_ff, visn_seff_ff, visn_seff_ff, ext_vmasks, None, 
                                                           text_ape, region_ape, t2r_rpe)
        
        visn_cross_ff = self.bert_visn_cross_FFN(visn_cross_att)
        text_cross_ff = self.bert_text_cross_FFN(text_cross_att)
        
        return visn_cross_ff, text_cross_ff

@ENCODER_REGISTRY.register()
class GPSCAEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        num_hidden_layers: int,
        bert_layers,
    ):
        super(GPSCAEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.layers = bert_layers
        self.region_box_embedding = nn.Linear(4, 512)
        self.text_box_embedding = nn.Linear(4, 512)
        self.fc_g = nn.Linear(64, 8)

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
            visn_feats = batched_inputs[kfg.ATT_FEATS]
            ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS]
            text_feats = batched_inputs[kfg.ATT_FEATS_WO_MASK]

            boxes = batched_inputs[kfg.POS_EMBED]
            n_text = 180
            text_boxes, region_boxes = boxes[:,:n_text], boxes[:,n_text:]
            region_ape = self.region_box_embedding(region_boxes)
            text_ape =  self.text_box_embedding(text_boxes)

            g = BoxRelationalEmbedding(boxes)
            b_s, n = g.shape[:2]
            g = self.fc_g(g.view(b_s * n, n, 64)).view(b_s, n, n, -1).permute(0, 3, 1, 2)
            geometric_attention = F.relu(g) # (b_s, h, n, n, d_k)
            t2t_rpe = geometric_attention[:, :, :n_text, :n_text]
            t2r_rpe = geometric_attention[:, :, :n_text, n_text:]
            r2r_rpe = geometric_attention[:, :, n_text:, n_text:]
            r2t_rpe = geometric_attention[:, :, n_text:, :n_text]
            
            gv_vmasks = torch.zeros((ext_vmasks.shape[0], 1, 1, 1), device=text_feats.device)
            ext_vmasks = torch.cat([gv_vmasks, ext_vmasks], dim=-1)
            
            for i, layer_module in enumerate(self.layers):
                visn_feats, text_feats = layer_module(visn_feats, text_feats, ext_vmasks, region_ape, text_ape, r2r_rpe, r2t_rpe, t2t_rpe, t2r_rpe)

            ret.update( {kfg.ATT_FEATS: visn_feats, kfg.ATT_FEATS_WO_MASK: text_feats, kfg.EXT_ATT_MASKS: ext_vmasks})
        return ret