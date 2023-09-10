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
class MutliCrossEntropy(nn.Module):
    @configurable
    def __init__(self):
        super(MutliCrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    @classmethod
    def from_config(cls, cfg):
        return {}

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, outputs_dict):
        ret  = {}
        logits = outputs_dict[kfg.G_LOGITS]
        targets = outputs_dict[kfg.G_TARGET_IDS]
        targets = targets.view(-1).long()

        logit1 = logits[-1].view(-1, logits[-1].shape[-1])
        logit2 = logits[-2].view(-1, logits[-2].shape[-1])
        logit3 = logits[-3].view(-1, logits[-3].shape[-1])
        loss1 = self.criterion(logit1, targets)
        loss2 = self.criterion(logit2, targets)
        loss3 = self.criterion(logit3, targets)
        loss = (3 * loss1 + loss2 + loss3) / 3
        
        ret.update({ 'CrossEntropy Loss(G)': loss })
            
        return ret

