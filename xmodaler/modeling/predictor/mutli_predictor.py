# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from torch import nn

from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import PREDICTOR_REGISTRY

__all__ = ["MutliPredictor"]

@PREDICTOR_REGISTRY.register()
class MutliPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,   # include <BOS>/<EOS>
        dropout: float
    ):
        super(MutliPredictor, self).__init__()
        self.output1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size), 
        )
        
        self.output2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size), 
        )
        
        self.output3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size), 
        )
        
    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.DECODER_DIM,
            "vocab_size": cfg.MODEL.VOCAB_SIZE,
            "dropout": cfg.MODEL.PRED_DROPOUT
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        hidden_states = batched_inputs[kfg.G_HIDDEN_STATES]

        if kfg.TIME_STEP in batched_inputs:
            output1 = self.output1(hidden_states[-1])
            return { kfg.G_LOGITS: output1 }
            
        output1 = self.output1(hidden_states[-1])
        output2 = self.output2(hidden_states[-2])
        output3 = self.output3(hidden_states[-3])
        return { kfg.G_LOGITS: [output1, output2, output3] }