# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li, Jianjie Luo
@contact: yehaoli.sysu@gmail.com, jianjieluo.sysu@gmail.com
"""
import itertools
import os
import copy
import torch
import pickle
import random
import h5py
import numpy as np
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, dict_as_tensor, boxes_to_locfeats
from ..build import DATASETS_REGISTRY

__all__ = ["MSCoCoVSDataset"]

@DATASETS_REGISTRY.register()
class MSCoCoVSDataset:
    @configurable
    def __init__(
        self,
        stage: str,
        anno_file: str,
        seq_per_img: int,
        max_feat_num: int,
        max_seq_len: int,
        region_feats_path: str,
        text_feats_path: str,
        global_feats_path: str,
        k: str,
    ):
        self.stage = stage
        self.anno_file = anno_file
        self.seq_per_img = seq_per_img
        self.max_feat_num = max_feat_num
        self.max_seq_len = max_seq_len
        self.k = k

        self.ctx = h5py.File(text_feats_path, 'r')
        self.region_f = h5py.File(region_feats_path, 'r')
        self.global_f = h5py.File(global_feats_path, 'r')
        
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
            "text_feats_path": cfg.DATALOADER.GRID_FEATS_PATH,
            "region_feats_path": cfg.DATALOADER.REGION_FEATS_PATH,
            "global_feats_path": cfg.DATALOADER.GV_FEAT_FILE,
            "k": cfg.DATALOADER.K_SAMPLE,
        }
        return ret

    def load_data(self, cfg):
        datalist = pickle.load(open(self.anno_file, 'rb'), encoding='bytes')
        # return datalist[:100]
        return datalist

    def __load(self, ctx, k):
        ctx_whole_f = ctx[f"{k}/whole/features"][:self.k]

        ctx_five_f = ctx[f"{k}/five/features"][:, :self.k]
        ctx_five_p = np.tile(np.arange(5)[:, None], (1, self.k))
        ctx_five_f = ctx_five_f.reshape((5*self.k, -1))
        ctx_five_p = ctx_five_p.reshape((5*self.k, ))

        ctx_nine_f = ctx[f"{k}/nine/features"][:, :self.k]
        ctx_nine_p = np.tile(np.arange(9)[:, None], (1, self.k))
        ctx_nine_f = ctx_nine_f.reshape((9*self.k, -1))
        ctx_nine_p = ctx_nine_p.reshape((9*self.k, ))

        return {
            "whole": {"features": ctx_whole_f},
            "five": {"features": ctx_five_f, "positions": ctx_five_p},
            "nine": {"features": ctx_nine_f, "positions": ctx_nine_p}
        }

    def parse_boxes(self, image_id):
        image_size = self.region_f['%s_size' % image_id][()][::-1]
        image_width, image_height = image_size[0], image_size[1]
        five_ratio = 0.6
        nine_ratio = 0.4

        text_boxes = []

        ########## orgin_boxes ###########
        orgin_box = np.array([0,0,image_width,image_height])
        text_boxes.append(orgin_box)

        ########## five_boxes ###########
        crop_height, crop_width = int(image_height*five_ratio), int(image_width*five_ratio)

        # tl = (0, 0, crop_height, crop_width)
        # tr = (0, image_width - crop_width, crop_height, image_width)
        # bl = (image_height - crop_height, 0, image_height, crop_width)
        # br = (image_height - crop_height, image_width - crop_width, image_height, image_width)

        t = (0, image_height - crop_height)
        b = (crop_height, image_height)
        l = (0, image_width - crop_width)
        r = (crop_width, image_width)
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        crop_bottom = crop_top + crop_height
        crop_right = crop_left + crop_width
        # center = (crop_top, crop_left, crop_bottom, crop_right)

        h, w = list(zip(t, b)), list(zip(l, r))
        five_arr = list(itertools.product(h, w))
        five_arr.append(((crop_top, crop_bottom),(crop_left, crop_right)))

        for s in five_arr:
            h, w = s
            x1, y1, x2, y2 = w[0], h[0], w[1], h[1]
            box = np.array([x1, y1, x2, y2])
            text_boxes.append(box)

        ########## nine_boxes ###########
        t = (0, int((0.5-nine_ratio/2)*image_height), int((1.0 - nine_ratio)*image_height))
        b = (int(nine_ratio*image_height), int((0.5+nine_ratio/2)*image_height), image_height)
        l = (0, int((0.5-nine_ratio/2)*image_width), int((1.0 - nine_ratio)*image_width))
        r = (int(nine_ratio*image_width), int((0.5+nine_ratio/2)*image_width), image_width)
        h, w = list(zip(t, b)), list(zip(l, r))

        nine_arr = list(itertools.product(h, w))
        for s in nine_arr:
            h, w = s
            x1, y1, x2, y2 = w[0], h[0], w[1], h[1]
            box = np.array([x1, y1, x2, y2])
            text_boxes.append(box)

        text_boxes = np.stack(text_boxes, axis=0)
        text_boxes = np.repeat(text_boxes, self.k, axis=0).astype('float32')

        global_box = np.array([[0,0,image_width,image_height]])
        region_boxes = self.region_f['%s_boxes' % image_id][()]
        region_boxes = region_boxes[0:self.max_feat_num].astype('float32')

        # boxes = np.concatenate([text_boxes, region_boxes], axis=0)
        boxes = np.concatenate([text_boxes, global_box, region_boxes], axis=0)
        size = np.concatenate([image_size[None], image_size[None]], axis=1)
        relative_boxes = boxes / size
        return relative_boxes.astype('float32')
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image_id = dataset_dict['image_id']
        ret = { kfg.IDS: image_id }

        data = self.__load(self.ctx, image_id)
        ctx_whole_f = data["whole"]["features"]
        ctx_five_f = data["five"]["features"]
        ctx_five_p = data["five"]["positions"]
        ctx_nine_f = data["nine"]["features"]
        ctx_nine_p = data["nine"]["positions"]

        ctx_whole_f = torch.FloatTensor(ctx_whole_f)
        ctx_five_f = torch.FloatTensor(ctx_five_f)
        ctx_five_p = torch.LongTensor(ctx_five_p)
        ctx_nine_f = torch.FloatTensor(ctx_nine_f)
        ctx_nine_p = torch.LongTensor(ctx_nine_p)
        ret.update( {kfg.CONTEXT: [ctx_whole_f, ctx_five_f, ctx_five_p, ctx_nine_f, ctx_nine_p]} )

        region_feats = self.region_f['%s_features' % image_id][()]
        region_feats = region_feats[0:self.max_feat_num].astype('float32')
        ret.update( {kfg.ATT_FEATS: region_feats} )
        # ret.update( {kfg.ATT_FEATS_WO_MASK: region_feats} )

        relative_boxes = self.parse_boxes(image_id)
        ret.update( {kfg.POS_EMBED: relative_boxes} )

        global_feats = self.global_f['%s_global' % image_id][()].astype('float32')
        ret.update( {kfg.GLOBAL_FEATS: global_feats} )

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