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
from ..embedding.build import EMBEDDING_REGISTRY
from ..embedding.visual_embed import VisualBaseEmbedding
from ..layers.positionwise_feedforward import PositionWiseFeedForward
from ..layers.multihead_attention import MultiHeadAttention, ScaledDotProductAttentionMemory

__all__ = ["MemoryAugmentedEncoder", "MeshedMemoryVisualEmbedding"]

@EMBEDDING_REGISTRY.register()
class MeshedMemoryVisualEmbedding(VisualBaseEmbedding):
    @configurable
    def __init__(
        self, 
        *,
        in_dim: int,
        out_dim: int,
        **kwargs
    ):
        kwargs['in_dim'] = in_dim
        kwargs['out_dim'] = out_dim
        super(MeshedMemoryVisualEmbedding, self).__init__(**kwargs)

    def forward(self, batched_inputs):
        feats = batched_inputs[kfg.ATT_FEATS]
        boxes = batched_inputs[kfg.ATT_FEATS_LOC] if kfg.ATT_FEATS_LOC in batched_inputs else None

        embeddings = self.embeddings(feats)
        if (self.embeddings_pos is not None) and (boxes is not None):
            embeddings_pos = self.embeddings_pos(boxes)
            embeddings = embeddings + embeddings_pos

        if self.embeddings_act is not None:
            embeddings = self.embeddings_act(embeddings)

        if self.embeddings_dropout is not None:
            embeddings = self.embeddings_dropout(embeddings)

        if self.embeddings_norm is not None:
            embeddings = self.embeddings_norm(embeddings)

        return { kfg.ATT_FEATS: embeddings }


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

        self.mhatt = MultiHeadAttention(  d_model=d_model, 
                                                d_k=d_k, 
                                                d_v=d_v, 
                                                num_head=num_head, 
                                                dropout=dropout, 
                                                attention_module=ScaledDotProductAttentionMemory,
                                                attention_module_kwargs={"m": num_memory}
                                                )

        self.pwff = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, queries, keys, values, attention_mask):
        att = self.mhatt(queries, keys, values, attention_mask)
        ff = self.pwff(att)
        return ff

@ENCODER_REGISTRY.register()
class MemoryAugmentedEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        input_dim: int, # out_dim of visual embedding
        d_model: int,   # hidden size
        num_layer: int, 
        num_att_head: int, 
        num_att_memory: int, # memory attention 
        d_ff: int, # feedforward size
        dropout: float,
        padding_idx: int
    ):
        super(MemoryAugmentedEncoder, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layer
        self.num_att_head = num_att_head
        self.num_att_memory = num_att_memory
        self.d_ff = d_ff
        self.padding_idx = padding_idx
        self.dropout = dropout

        # NOTE: Do this in `MeshedMemoryVisualEmbedding`
        # # encoder input layer
        # self.fc = nn.Linear(self.input_dim, self.d_model)
        # self.dropout = nn.Dropout(p=self.dropout) if self.dropout > 0. else None
        # self.layer_norm = nn.LayerNorm(self.d_model)

        # encoder hidden layers
        self.layers = nn.ModuleList([EncoderLayer(  d_model=self.d_model,  
                                                    num_head=self.num_att_head, 
                                                    num_memory=self.num_att_memory,
                                                    d_ff=self.d_ff, 
                                                    dropout=dropout    
                                                    )
                                     for _ in range(self.num_layers)])

    @classmethod
    def from_config(cls, cfg):

        return {
            "input_dim": cfg.MODEL.VISUAL_EMBED.OUT_DIM,
            "d_model": cfg.MODEL.MESHEDMEORY.ENCODER.DIM_MODEL,
            "num_layer": cfg.MODEL.MESHEDMEORY.ENCODER.NUM_LAYER,
            "num_att_head": cfg.MODEL.MESHEDMEORY.ENCODER.NUM_ATT_HEAD,
            "num_att_memory": cfg.MODEL.MESHEDMEORY.ENCODER.NUM_ATT_MEMORY,
            "d_ff": cfg.MODEL.MESHEDMEORY.ENCODER.DIM_FEEDFORWARD,
            "dropout": cfg.MODEL.MESHEDMEORY.ENCODER.DROPOUT,
            "padding_idx": 0 # default
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.MESHEDMEORY = CN()
        
        cfg.MODEL.MESHEDMEORY.ENCODER = CN()

        cfg.MODEL.MESHEDMEORY.ENCODER.DIM_MODEL = 512
        cfg.MODEL.MESHEDMEORY.ENCODER.NUM_LAYER = 3
        cfg.MODEL.MESHEDMEORY.ENCODER.DROPOUT = 0.1
        cfg.MODEL.MESHEDMEORY.ENCODER.NUM_ATT_HEAD = 8
        cfg.MODEL.MESHEDMEORY.ENCODER.NUM_ATT_MEMORY = 40
        cfg.MODEL.MESHEDMEORY.ENCODER.DIM_FEEDFORWARD = 2048

    def _get_global_feat(self, feats, masks):
        if masks is None:
            global_feats = torch.mean(feats, 1)
        else:
            feats_masks = feats * masks.unsqueeze(-1)
            masks_sum = masks.sum(-1)
            global_feats = feats_masks.sum(1) / masks_sum.unsqueeze(-1)
        return global_feats

    def forward(self, batched_inputs, mode=None):
        ret = {}
        if mode == None or mode == 'v':
            if kfg.ATT_FEATS in batched_inputs:
                att_feats = batched_inputs[kfg.ATT_FEATS]
                att_masks = batched_inputs[kfg.ATT_MASKS]
                attention_mask = (att_masks == 0)
            else:
                att_feats = batched_inputs[kfg.ATT_FEATS_WO_MASK]
                attention_mask = None

            outs = []
            out = att_feats
            for l in self.layers:
                out = l(out, out, out, attention_mask)
                outs.append(out.unsqueeze(1))

            outs = torch.cat(outs, 1) # [batch, num_layer, seq_len, d_model]

            if kfg.ATT_FEATS in batched_inputs:
                ret.update(
                    {
                        kfg.ATT_FEATS: outs,
                        kfg.ATT_MASKS: attention_mask
                    }
                )
            else:
                ret.update( {kfg.ATT_FEATS_WO_MASK: outs })
        return ret