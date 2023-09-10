# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.modeling.embedding.position_embedding import SinusoidEncoding, sinusoid_encoding_table
from ..layers.create_act import get_act_layer
from .build import EMBEDDING_REGISTRY

__all__ = ["VisualContextEmbedding"]

@EMBEDDING_REGISTRY.register()
class VisualContextEmbedding(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        f_obj,
        f_out,
        f_vis,
        f_txt,
        activation_name,
        dropout,
    ):
        super(VisualContextEmbedding, self).__init__()

        # for objects O
        self.obj_mlp1 = nn.Sequential(
            nn.LayerNorm(f_obj), nn.Linear(f_obj, f_out), nn.Dropout(p=dropout)
        )
        self.obj_mlp2 = nn.Sequential(
            nn.LayerNorm(f_vis), nn.Linear(f_vis, f_out), nn.Dropout(p=dropout)
        )

        # for txt_ctx
        # self.txt_keys = ("whole", "five", "nine")
        # for k in self.txt_keys:
        #     mlp1 = nn.Sequential(
        #         nn.LayerNorm(f_txt), nn.Linear(f_txt, f_out), nn.Dropout(p=dropout)
        #     )
        #     mlp2 = nn.Sequential(
        #         nn.LayerNorm(f_vis), nn.Linear(f_vis, f_out), nn.Dropout(p=dropout)
        #     )
        #     setattr(self, f"txt_mlp1_{k}", mlp1)
        #     setattr(self, f"txt_mlp2_{k}", mlp2)

        #     if k == "whole":
        #         num_embeddings = 1
        #     elif k == "five":
        #         num_embeddings = 5
        #     elif k == "nine":
        #         num_embeddings = 9
        #     else:
        #         raise KeyError
                   
        #     pos = nn.Embedding.from_pretrained(
        #         sinusoid_encoding_table(num_embeddings, f_out), freeze=True
        #     )
            
        #     setattr(self, f"txt_pos_{k}", pos)

    @classmethod
    def from_config(cls, cfg):
        in_dim = cfg.MODEL.VISUAL_EMBED.IN_DIM
        out_dim = cfg.MODEL.VISUAL_EMBED.OUT_DIM
        activation_name = (cfg.MODEL.VISUAL_EMBED.ACTIVATION).lower()
        dropout = cfg.MODEL.VISUAL_EMBED.DROPOUT

        return {
            'f_obj': in_dim,
            'f_out': out_dim,
            'f_vis': 768,
            # 'f_vis': 640,
            'f_txt': 512,
            'activation_name': activation_name,
            'dropout': dropout,
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        vis_ctx = batched_inputs[kfg.GLOBAL_FEATS]
        # obj = batched_inputs[kfg.ATT_FEATS]
        obj = batched_inputs[kfg.ATT_FEATS_WO_MASK]
        # txt_ctx = batched_inputs[kfg.CONTEXT]
        # ext_mask = batched_inputs[kfg.EXT_ATT_MASKS]

        img = vis_ctx[:, None, :]
        # embed = []

        # object O
        obj_embed = self.obj_mlp1(obj) + self.obj_mlp2(img)
        # embed.append(obj_embed)

        # ctx T
        # for k in self.txt_keys:
        #     pos_k = txt_ctx[k]["pos"]
        #     embed_k = txt_ctx[k]["embed"]
        #     mlp1 = getattr(self, f"txt_mlp1_{k}")
        #     mlp2 = getattr(self, f"txt_mlp2_{k}")
        #     mlp_pos = getattr(self, f"txt_pos_{k}")
        #     embed_k = mlp1(embed_k) + mlp2(img) + mlp_pos(pos_k)
        #     embed.append(embed_k)
        
        # mix_feats = torch.cat(embed, dim=1)
        # bs, n, _ = mix_feats.shape
        # exp_mask = torch.zeros((bs, 1, 1, n), device=mix_feats.device)
        # exp_mask[:,:,:,:obj.size(1)] = ext_mask
        return { 
            # kfg.ATT_FEATS : obj_embed,
            kfg.ATT_FEATS_WO_MASK : obj_embed,
            # kfg.EXT_ATT_MASKS : exp_mask
            # kfg.ATT_FEATS_WO_MASK: mix_feats
        }