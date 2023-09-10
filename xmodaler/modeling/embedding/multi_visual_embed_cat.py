# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from math import sqrt
import torch
from torch import nn
from einops import rearrange, repeat
from torch.nn.functional import interpolate

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.modeling.embedding.position_embedding import PolarRPE
from ..layers.create_act import get_act_layer
from .build import EMBEDDING_REGISTRY

__all__ = ["MultiVisualEmbedding"]

@EMBEDDING_REGISTRY.register()
class MultiVisualEmbedding(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        in_dim,
        out_dim,
        activation_name,
        dropout,
    ):
        super(MultiVisualEmbedding, self).__init__()

        self.mix_embeddings = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            get_act_layer(activation_name)(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )

        # self.rpe = PolarRPE(k=3, h=8, d_k=64, d_r=256, window_size=(9, 9))

    @classmethod
    def from_config(cls, cfg):
        in_dim = cfg.MODEL.VISUAL_EMBED.IN_DIM
        out_dim = cfg.MODEL.VISUAL_EMBED.OUT_DIM
        activation_name = (cfg.MODEL.VISUAL_EMBED.ACTIVATION).lower()
        dropout = cfg.MODEL.VISUAL_EMBED.DROPOUT

        return {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'activation_name': activation_name,
            'dropout': dropout,
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        vit_feats = batched_inputs[kfg.ATT_FEATS]
        res_feats = batched_inputs[kfg.ATT_FEATS_WO_MASK]
        global_feats = batched_inputs[kfg.GLOBAL_FEATS]

        sh = sw = int(sqrt(vit_feats.shape[1]))
        lh = lw = int(sqrt(res_feats.shape[1]))
        vit_feats = rearrange(vit_feats, 'b (h w) d -> b d h w', h=sh, w=sw)  # (b_s, d, sh, sw)
        vit_feats = interpolate(vit_feats, size=(lh, lw), mode="nearest")
        vit_feats = rearrange(vit_feats, 'b d h w -> b (h w) d', h=lh, w=lw)  # (b_s, n, d)

        global_feats = repeat(global_feats, 'b d -> b n d', n=lh*lw)
        mix_feats = torch.cat([vit_feats, res_feats, global_feats], dim=-1)
        mix_feats = self.mix_embeddings(mix_feats)

        # rpe = self.rpe(mix_feats.size(0))

        batched_inputs.pop(kfg.ATT_FEATS)
        batched_inputs.pop(kfg.ATT_MASKS)
        return { 
            kfg.ATT_FEATS_WO_MASK : mix_feats,
            # kfg.POS_EMBED: rpe
        }