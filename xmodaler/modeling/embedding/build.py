# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from xmodaler.utils.registry import Registry

EMBEDDING_REGISTRY = Registry("EMBEDDING")
EMBEDDING_REGISTRY.__doc__ = """
Registry for embedding
"""

def build_embeddings(cfg, name):
    embeddings = EMBEDDING_REGISTRY.get(name)(cfg)
    return embeddings

def add_embedding_config(cfg, tmp_cfg):
    if len(tmp_cfg.MODEL.VISUAL_EMBED.NAME) > 0:
        EMBEDDING_REGISTRY.get(tmp_cfg.MODEL.VISUAL_EMBED.NAME).add_config(cfg)