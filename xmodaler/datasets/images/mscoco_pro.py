# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li, Jianjie Luo
@contact: yehaoli.sysu@gmail.com, jianjieluo.sysu@gmail.com
"""
import os
import copy
import pickle
import random
import h5py
import numpy as np
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, dict_as_tensor, boxes_to_locfeats
from ..build import DATASETS_REGISTRY

__all__ = ["MSCoCoProDataset"]

@DATASETS_REGISTRY.register()
class MSCoCoProDataset:
    @configurable
    def __init__(
        self,
        stage: str,
        anno_file: str,
        seq_per_img: int,
        max_feat_num: int,
        max_seq_len: int,
        grid_feats_path: str,
        region_feats_path: str,
        global_feats_path: str,
    ):
        self.stage = stage
        self.anno_file = anno_file
        self.seq_per_img = seq_per_img
        self.max_feat_num = max_feat_num
        self.max_seq_len = max_seq_len
        self.grid_feats_path = grid_feats_path
        self.region_feats_path = region_feats_path
        self.global_feats_path = global_feats_path

        if len(self.grid_feats_path) > 0:
            self.grid_f = h5py.File(self.grid_feats_path, 'r')

        if len(self.region_feats_path) > 0:
            self.region_f = h5py.File(self.region_feats_path, 'r')

        if len(self.global_feats_path) > 0:
            self.global_f = h5py.File(self.global_feats_path, 'r')
        
    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ann_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "mscoco_caption_anno_train.pkl"),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "mscoco_caption_anno_val.pkl"),
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "mscoco_caption_anno_test.pkl")
        }
        ret = {
            "stage": stage,
            "anno_file": ann_files[stage],
            "seq_per_img": cfg.DATALOADER.SEQ_PER_SAMPLE,
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
            "grid_feats_path": cfg.DATALOADER.GRID_FEATS_PATH,
            "region_feats_path": cfg.DATALOADER.REGION_FEATS_PATH,
            "global_feats_path": cfg.DATALOADER.GV_FEAT_FILE,
        }
        return ret

    # def load_data(self, cfg):
    #     datalist = pickle.load(open(self.anno_file, 'rb'), encoding='bytes')
    #     if self.stage != 'train':
    #         return datalist
    #     # return datalist[:100]
    #     expand_datalist = []
    #     for data in datalist:
    #         for i in range(5):
    #             expand_datalist.append(
    #                 {
    #                     'image_id': data['image_id'],
    #                     'tokens_ids': data['tokens_ids'][i,None],
    #                     'target_ids': data['target_ids'][i,None]
    #                 }
    #             )
    #     return expand_datalist

    def load_data(self, cfg):
        datalist = pickle.load(open(self.anno_file, 'rb'), encoding='bytes')
        # return datalist[:100]
        return datalist
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image_id = dataset_dict['image_id']
        ret = { kfg.IDS: image_id }

        if len(self.grid_feats_path) > 0:
            grid_feats = self.grid_f['%s_features' % image_id][()]
            grid_feats = grid_feats.astype('float32')
            ret.update( {kfg.ATT_FEATS_WO_MASK: grid_feats} )

        if len(self.global_feats_path) > 0:
            global_feats = self.global_f['%s_global' % image_id][()].astype('float32')
            ret.update( {kfg.GLOBAL_FEATS: global_feats} )

        if len(self.region_feats_path) > 0:
            region_feats = self.region_f['%s_features' % image_id][()]
            region_feats = region_feats[0:self.max_feat_num].astype('float32')
            ret.update( {kfg.ATT_FEATS: region_feats} )
            
        if self.stage != 'train':
            g_tokens_type = np.ones((self.max_seq_len,), dtype=np.int64)
            ret.update({ kfg.G_TOKENS_TYPE: g_tokens_type })
            dict_as_tensor(ret)
            return ret
        
        sent_num = len(dataset_dict['tokens_ids'])
        if sent_num >= self.seq_per_img:
            selects = random.sample(range(sent_num), self.seq_per_img)
        else:
            selects = random.choices(range(sent_num), k = (self.seq_per_img - sent_num))
            selects += list(range(sent_num))

        tokens_ids = [ dataset_dict['tokens_ids'][i,:].astype(np.int64) for i in selects ]
        target_ids = [ dataset_dict['target_ids'][i,:].astype(np.int64) for i in selects ]
        g_tokens_type = [ np.ones((len(dataset_dict['tokens_ids'][i,:]), ), dtype=np.int64) for i in selects ]

        ret.update({
            kfg.SEQ_PER_SAMPLE: self.seq_per_img,
            kfg.G_TOKENS_IDS: tokens_ids,
            kfg.G_TARGET_IDS: target_ids,
            kfg.G_TOKENS_TYPE: g_tokens_type,
        })

        dict_as_tensor(ret)
        return ret