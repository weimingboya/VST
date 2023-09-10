# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import math
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import kfg
from ..layers.create_act import get_act_layer
from .build import EMBEDDING_REGISTRY
from .position_embedding import GridPESine, GridPELearned

__all__ = ["ViusualTokenEmbedding"]

@EMBEDDING_REGISTRY.register()
class ViusualTokenEmbedding(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        dim: int,
        vocab_size: int, # include <BOS>/<EOS>
        **kwargs
    ):
        super(ViusualTokenEmbedding, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, dim)
        self.embeddings_act = kwargs.pop("embeddings_act", None)
        self.embeddings_norm = kwargs.pop("embeddings_norm", None)
        self.embeddings_dropout = kwargs.pop("embeddings_dropout", None)
        self.embeddings_pos = kwargs.pop("embeddings_pos", None)

    @classmethod
    def from_config(cls, cfg):
        kwargs = {
            "dim": cfg.MODEL.VISUAL_EMBED.DIM, 
            "vocab_size": cfg.MODEL.VISUAL_EMBED.VOCAB_SIZE
        }

        activation_name = (cfg.MODEL.VISUAL_EMBED.ACTIVATION).lower()
        if activation_name != "none":
            activation = get_act_layer(activation_name)
            assert activation is not None

            act_kwargs = {}
            if activation_name in { "elu", "celu" }:
                act_kwargs["alpha"] = cfg.MODEL.VISUAL_EMBED.ELU_ALPHA
            embeddings_act = activation(**act_kwargs)
            kwargs['embeddings_act'] = embeddings_act

        if cfg.MODEL.VISUAL_EMBED.DROPOUT > 0:
            embeddings_dropout = nn.Dropout(cfg.MODEL.VISUAL_EMBED.DROPOUT)
            kwargs['embeddings_dropout'] = embeddings_dropout

        if cfg.MODEL.VISUAL_EMBED.USE_NORM:
            embeddings_norm = nn.LayerNorm(cfg.MODEL.VISUAL_EMBED.DIM)
            kwargs['embeddings_norm'] = embeddings_norm

        if (cfg.MODEL.VISUAL_EMBED.POSITION).lower() != 'none':
            if cfg.MODEL.VISUAL_EMBED.POSITION == 'GridPESine':
                embeddings_pos = GridPESine(num_pos_feats=cfg.MODEL.VISUAL_EMBED.DIM // 2, normalize=True)
            elif cfg.MODEL.VISUAL_EMBED.POSITION == 'GridPELearned':
                embeddings_pos = GridPELearned(num_pos_feats=cfg.MODEL.VISUAL_EMBED.DIM // 2)
            kwargs['embeddings_pos'] = embeddings_pos

        return kwargs

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.VISUAL_EMBED.DIM = 512
        cfg.MODEL.VISUAL_EMBED.VOCAB_SIZE = 8196
        cfg.MODEL.VISUAL_EMBED.POSITION = 'none'

    def forward(self, batched_inputs):
        feats = batched_inputs[kfg.ATT_FEATS_WO_MASK].long()

        embeddings = self.embeddings(feats)

        if self.embeddings_pos is not None:
            bs, num_tokens = feats.shape[:2]
            grid_size = int(math.sqrt(num_tokens))
            position_embeddings = self.embeddings_pos(feats.view(bs, grid_size, grid_size, -1))
            embeddings = embeddings + position_embeddings

        if self.embeddings_act is not None:
            embeddings = self.embeddings_act(embeddings)

        if self.embeddings_norm is not None:
            embeddings = self.embeddings_norm(embeddings)

        if self.embeddings_dropout is not None:
            embeddings = self.embeddings_dropout(embeddings)
            
        return { kfg.ATT_FEATS_WO_MASK: embeddings}