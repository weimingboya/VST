# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from .build import build_embeddings, add_embedding_config
from .token_embed import TokenBaseEmbedding
from .visual_embed import VisualBaseEmbedding, VisualIdentityEmbedding
from .visual_embed_conv import TDConvEDVisualBaseEmbedding
from .visual_grid_embed import VisualGridEmbedding
from .dual_visual_embed import DualVisualEmbedding
from .visual_token_embed import ViusualTokenEmbedding
from .visual_embed_kd import VisualBaseEmbeddingWithKD
from .visual_context_embed import VisualContextEmbedding
from .multi_visual_embed_pe import MultiVisualEmbedding
# from .multi_visual_embed_fusion import MultiVisualEmbedding

__all__ = list(globals().keys())