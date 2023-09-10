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
from xmodaler.modeling.embedding.position_embedding import GridPELearned, GridPESine, SinusoidEncoding, sinusoid_encoding_table
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

        # self.pe = GridPELearned(f_out // 2)
        # self.pe = GridPESine(f_out // 2)
        # self.pe = nn.Linear(4, 512)

        # for txt_ctx
        self.txt_keys = ("whole", "five", "nine")
        # self.txt_keys = ("five", "nine")
        for k in self.txt_keys:
            mlp1 = nn.Sequential(
                nn.LayerNorm(f_txt), nn.Linear(f_txt, f_out), nn.Dropout(p=dropout)
            )
            # mlp2 = nn.Sequential(
            #     nn.LayerNorm(f_vis), nn.Linear(f_vis, f_out), nn.Dropout(p=dropout)
            # )
            setattr(self, f"txt_mlp1_{k}", mlp1)
            # setattr(self, f"txt_mlp2_{k}", mlp2)

            # if k == "whole":
            #     num_embeddings = 1
            # elif k == "five":
            #     num_embeddings = 5
            # elif k == "nine":
            #     num_embeddings = 9
            # else:
            #     raise KeyError
                   
            # pos = nn.Embedding.from_pretrained(
            #     sinusoid_encoding_table(num_embeddings, f_out), freeze=True
            # )
            
            # setattr(self, f"txt_pos_{k}", pos)

    @classmethod
    def from_config(cls, cfg):
        in_dim = cfg.MODEL.VISUAL_EMBED.IN_DIM
        out_dim = cfg.MODEL.VISUAL_EMBED.OUT_DIM
        activation_name = (cfg.MODEL.VISUAL_EMBED.ACTIVATION).lower()
        dropout = cfg.MODEL.VISUAL_EMBED.DROPOUT

        return {
            'f_obj': in_dim,
            'f_out': out_dim,
            # 'f_vis': 768,
            'f_vis': 512,
            'f_txt': 512,
            'activation_name': activation_name,
            'dropout': dropout,
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        gv_feat = batched_inputs[kfg.GLOBAL_FEATS][:, None, :]
        # obj = batched_inputs[kfg.ATT_FEATS_WO_MASK]
        # boxes = batched_inputs[kfg.POS_EMBED]
        obj = batched_inputs[kfg.ATT_FEATS]
        txt_ctx = batched_inputs[kfg.CONTEXT]
        # batched_inputs.pop(kfg.CONTEXT)

        embed = []

        # object O
        obj_embed = self.obj_mlp1(obj)
        gv_embed = self.obj_mlp2(gv_feat)
        obj_embed = torch.cat([gv_embed, obj_embed], dim=1)
        # obj_embed = self.obj_mlp1(obj) + self.obj_mlp2(gv_feat)

        
        # position encoding
        # bs, n = obj_embed.shape[:2]
        # grid_size = int(math.sqrt(n))
        # pe = self.pe(obj_embed.transpose(-1,-2).view(bs, -1, grid_size, grid_size))
        # pe = pe.permute(0, 2, 3, 1).view(bs, grid_size * grid_size, -1)
        # pe = self.pe(obj_embed.view(bs, grid_size, grid_size, -1))
        # obj_embed = obj_embed + pe

        # pe = self.pe(boxes)
        # obj_embed = obj_embed + pe

        # ctx T
        for k in self.txt_keys:
            # pos_k = txt_ctx[k]["pos"]
            embed_k = txt_ctx[k]["embed"]
            mlp1 = getattr(self, f"txt_mlp1_{k}")
            # mlp2 = getattr(self, f"txt_mlp2_{k}")
            # mlp_pos = getattr(self, f"txt_pos_{k}")
            # embed_k = mlp1(embed_k) + mlp2(vis_ctx) + mlp_pos(pos_k)
            # embed_k = mlp1(embed_k) + mlp_pos(pos_k)
            # embed_k = mlp1(embed_k) + mlp2(gv_feat)
            embed_k = mlp1(embed_k)
            embed.append(embed_k)
            
        txt_embed = torch.cat(embed, dim=1) # [81,12,5*12,9*12]
        # ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS]
        # mix_feats_arr = torch.cat([obj_embed, txt_embed], dim=1)
        # text_vmasks = torch.zeros((txt_embed.shape[0], 1, 1, txt_embed.shape[1]), device=txt_embed.device)
        # ext_vmasks = torch.cat([ext_vmasks, text_vmasks], dim=-1)
        # return {kfg.ATT_FEATS: mix_feats_arr, kfg.EXT_ATT_MASKS: ext_vmasks}
        return { 
            kfg.ATT_FEATS : obj_embed,
            # kfg.GLOBAL_FEATS : gv_embed,
            kfg.ATT_FEATS_WO_MASK : txt_embed,
        }