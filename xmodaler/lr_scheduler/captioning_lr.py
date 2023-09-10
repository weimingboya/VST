# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from xmodaler.config import configurable
from .build import LR_SCHEDULER_REGISTRY
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler

@LR_SCHEDULER_REGISTRY.register()
class CaptioningLR(_LRScheduler):
    @configurable
    def __init__(
        self, 
        *,
        optimizer,
        milestones,
        data_size: int,
        gamma=0.2,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        self.milestones = milestones
        self.data_size = data_size
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters

        super(CaptioningLR, self).__init__(optimizer, last_epoch=last_epoch)

    @classmethod
    def from_config(cls, cfg, optimizer, data_size):
        return {
            "optimizer": optimizer,
            "data_size": data_size,
            "milestones": cfg.LR_SCHEDULER.STEPS,
            "gamma": cfg.LR_SCHEDULER.GAMMA,
            "warmup_factor": cfg.LR_SCHEDULER.WARMUP_FACTOR,
            "warmup_iters": cfg.LR_SCHEDULER.WARMUP,
            "last_epoch": -1
        }

    def get_lr(self):
        warmup_factor = 1
        last_iter = self.last_epoch // self.data_size + 1
        if last_iter < self.warmup_iters:
            alpha = last_iter / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha

        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, last_iter)
            for base_lr in self.base_lrs
        ]
            