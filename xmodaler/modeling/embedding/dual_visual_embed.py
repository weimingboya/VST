# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import math
from einops import rearrange
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.modeling.embedding.position_embedding import GridPELearned, GridPESine, NNEmbeddingEncoding, SinusoidEncoding
from ..layers.create_act import get_act_layer
from .build import EMBEDDING_REGISTRY

__all__ = ["DualVisualEmbedding"]

@EMBEDDING_REGISTRY.register()
class DualVisualEmbedding(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        grid_in_dim,
        region_in_dim,
        out_dim,
        activation_name,
        dropout,
    ):
        super(DualVisualEmbedding, self).__init__()

        self.grid_embeddings = nn.Sequential(
            nn.Linear(grid_in_dim, out_dim),
            get_act_layer(activation_name)(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )

        self.region_embeddings = nn.Sequential(
            nn.Linear(region_in_dim, out_dim),
            get_act_layer(activation_name)(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )

        # self.time_pe = SinusoidEncoding(dim=out_dim, max_len=32)
        # self.space_pe = GridPESine(num_pos_feats=out_dim//2)

    @classmethod
    def from_config(cls, cfg):
        grid_in_dim = cfg.MODEL.VISUAL_EMBED.GRID_IN_DIM
        region_in_dim = cfg.MODEL.VISUAL_EMBED.REGION_IN_DIM
        out_dim = cfg.MODEL.VISUAL_EMBED.OUT_DIM
        activation_name = (cfg.MODEL.VISUAL_EMBED.ACTIVATION).lower()
        dropout = cfg.MODEL.VISUAL_EMBED.DROPOUT

        return {
            'grid_in_dim': grid_in_dim,
            'region_in_dim': region_in_dim,
            'out_dim': out_dim,
            'activation_name': activation_name,
            'dropout': dropout,
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        grid_feats = batched_inputs[kfg.ATT_FEATS_WO_MASK] # (bs, t, n, d)
        grid_feats = rearrange(grid_feats, "B T N D -> (B T) N D")
        region_feats = batched_inputs[kfg.ATT_FEATS]

        grid_embeddings = self.grid_embeddings(grid_feats)
        region_embeddings = self.region_embeddings(region_feats)

        # time_pe = self.time_pe(region_embeddings)
        # region_embeddings = region_embeddings + time_pe

        # bs, n = grid_embeddings.shape[:2]
        # grid_size = int(math.sqrt(n))
        # space_pe = self.space_pe(grid_embeddings.view(bs, grid_size, grid_size, -1))
        # space_pe = space_pe.permute(0, 2, 3, 1).view(bs, grid_size * grid_size, -1)
        # grid_embeddings = grid_embeddings + space_pe

        return { 
            kfg.ATT_FEATS : region_embeddings,
            kfg.ATT_FEATS_WO_MASK : grid_embeddings
        }