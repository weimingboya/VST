# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import kfg
from ..layers.create_act import get_act_layer
from .build import EMBEDDING_REGISTRY

__all__ = ["VisualBaseEmbeddingWithKD"]

@EMBEDDING_REGISTRY.register()
class VisualBaseEmbeddingWithKD(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        in_dim: int,
        out_dim: int,
        tearch_model,
        **kwargs
    ):
        super(VisualBaseEmbeddingWithKD, self).__init__()
        self.embeddings = nn.Linear(in_dim, out_dim)
        self.embeddings_act = kwargs.pop("embeddings_act", None)
        self.embeddings_norm = kwargs.pop("embeddings_norm", None)
        self.embeddings_dropout = kwargs.pop("embeddings_dropout", None)
        self.embeddings_pos = kwargs.pop('embeddings_pos', None)
        self.tearch_model = tearch_model

    @classmethod
    def from_config(cls, cfg):
        from xmodaler.modeling import build_model
        from xmodaler.config import get_cfg
        from xmodaler.modeling import add_config
        from xmodaler.checkpoint import XmodalerCheckpointer
        config_file = 'configs/image_caption/transformer/CLIP-ViT-L_transformer.yaml'
        t_cfg = get_cfg() # obtain X-modaler's default config
        tmp_cfg = t_cfg.load_from_file_tmp(config_file) # load custom config
        add_config(t_cfg, tmp_cfg) # combining default and custom configs
        t_cfg.merge_from_file(config_file) # load values from a file
        model = build_model(t_cfg)
        # checkpoint = torch.load('checkpoints/transformer/ViT-L-14-2/model_Epoch_00017_Iter_0192575.pth', map_location=cfg.MODEL.DEVICE)
        # tearch_model = model.load_state_dict(checkpoint['model'], strict=False)
        checkpointer = XmodalerCheckpointer(model)
        checkpointer.resume_or_load('checkpoints/transformer/ViT-L-14-2/model_Epoch_00017_Iter_0192575.pth', resume=True) # load a checkpoint
        tearch_model = checkpointer.model

        kwargs = {
            "in_dim": cfg.MODEL.VISUAL_EMBED.IN_DIM,
            "out_dim": cfg.MODEL.VISUAL_EMBED.OUT_DIM,
            "tearch_model": tearch_model
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
            embeddings_norm = nn.LayerNorm(cfg.MODEL.VISUAL_EMBED.OUT_DIM)
            kwargs['embeddings_norm'] = embeddings_norm

        if cfg.MODEL.VISUAL_EMBED.LOCATION_SIZE > 0:
            embeddings_pos = nn.Linear(5, cfg.MODEL.VISUAL_EMBED.OUT_DIM)
            kwargs['embeddings_pos'] = embeddings_pos

        return kwargs

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        feats = batched_inputs[kfg.ATT_FEATS] if kfg.ATT_FEATS in batched_inputs else batched_inputs[kfg.ATT_FEATS_WO_MASK]
        boxes = batched_inputs[kfg.ATT_FEATS_LOC] if kfg.ATT_FEATS_LOC in batched_inputs else None

        feats = batched_inputs[kfg.ATT_FEATS]
        batched_inputs.pop(kfg.ATT_FEATS)
        batched_inputs.pop(kfg.ATT_MASKS)
        res = {}

        self.tearch_model.eval()
        if kfg.G_TARGET_IDS in batched_inputs:
            with torch.no_grad():
                out = self.tearch_model(batched_inputs)
                res.update({ 'T_G_LOGITS': out[kfg.G_LOGITS] })

        batched_inputs[kfg.ATT_FEATS_WO_MASK] = feats

        embeddings = self.embeddings(feats)
        if (self.embeddings_pos is not None) and (boxes is not None):
            embeddings_pos = self.embeddings_pos(boxes)
            embeddings = embeddings + embeddings_pos

        if self.embeddings_act is not None:
            embeddings = self.embeddings_act(embeddings)

        if self.embeddings_norm is not None:
            embeddings = self.embeddings_norm(embeddings)

        if self.embeddings_dropout is not None:
            embeddings = self.embeddings_dropout(embeddings)

        feat_key = kfg.ATT_FEATS if kfg.ATT_FEATS in batched_inputs else kfg.ATT_FEATS_WO_MASK
        res.update({ feat_key : embeddings })
        return res