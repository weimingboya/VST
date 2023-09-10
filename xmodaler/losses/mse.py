# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
import torch.nn as nn
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class MSE(nn.Module):
    @configurable
    def __init__(self):
        super(MSE, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')

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
        loss = self.criterion(logits, t_logits) * 0.05
        ret.update({ 'MSE Loss(G)': loss })
            
        return ret

