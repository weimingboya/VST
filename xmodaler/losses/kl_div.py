# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
import torch.nn as nn
from xmodaler.config import configurable
from xmodaler.config import kfg
import torch.nn.functional as F
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class KL_div(nn.Module):
    @configurable
    def __init__(self):
        super(KL_div, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    @classmethod
    def from_config(cls, cfg):
        return {}

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, outputs_dict):
        ret  = {}
        t_logits = outputs_dict['T_G_LOGITS']
        logits = outputs_dict[kfg.G_LOGITS]
        loss = self.criterion(F.log_softmax(logits/10, dim=-1), F.softmax(t_logits/10, dim=-1))
        ret.update({ 'KL Loss(G)': loss })
            
        return ret

