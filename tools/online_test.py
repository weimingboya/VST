import copy
import itertools
import json
import torch
import h5py
import numpy as np
from pycocotools.coco import COCO
import tqdm
from xmodaler.config import get_cfg, configurable, kfg
from xmodaler.datasets.images.mscoco_vs import MSCoCoVSDataset
from xmodaler.modeling.meta_arch.ensemble import Ensemble
from xmodaler.utils import comm
from xmodaler.modeling import add_config
from xmodaler.modeling import build_model
from xmodaler.checkpoint import XmodalerCheckpointer
from xmodaler.datasets.build import build_dataset_mapper, trivial_batch_collator
from xmodaler.functional import read_np, dict_as_tensor, boxes_to_locfeats

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class MSCoCoTestDataset():
    @configurable
    def __init__(
        self,
        anno_file: str,
        max_feat_num: int,
        max_seq_len: int,
        region_feats_path: str,
        text_feats_path: str,
        global_feats_path: str,
        k: str,
    ):
        self.max_feat_num = max_feat_num
        self.max_seq_len = max_seq_len
        self.k = k
        
        coco_dataset = COCO(anno_file)
        self.dataset_dict = []
        for img_id, _ in coco_dataset.imgs.items():
            self.dataset_dict.append(img_id)

        self.ctx = h5py.File(text_feats_path, 'r')
        self.region_f = h5py.File(region_feats_path, 'r')
        self.global_f = h5py.File(global_feats_path, 'r')
        
    @classmethod
    def from_config(cls, cfg, split: str = "test"):
        ret = {
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
            "k": cfg.DATALOADER.K_SAMPLE,
        }
        if (split=='test'):
            ret.update({
                "anno_file": 'datasets/mscoco_dataset/image_info_test2014.json',
                "text_feats_path": '/home/xys/code/captioning/examples/Xmodal-Ctx/ctx/outputs/retrieved_captions/txt_ctx.hdf5',
                "region_feats_path": 'datasets/mscoco_dataset/features/COCO2014_VinVL_TEST.hdf5',
                "global_feats_path": 'datasets/mscoco_dataset/features/COCO2014_ViT-B-32_GLOBAL_TEST.hdf5',
            })
        else:
            ret.update({
                "anno_file": '/home/public2/caption/coco2014/annotations/captions_val2014.json',
                "text_feats_path": 'datasets/mscoco_dataset/features/txt_ctx.hdf5',
                "region_feats_path": 'datasets/mscoco_dataset/features/COCO2014_VinVL.hdf5',
                "global_feats_path": 'datasets/mscoco_dataset/features/COCO2014_ViT-B-32_GLOBAL.hdf5',
            })
            
        return ret
    
    def __len__(self):
        return len(self.dataset_dict)
    
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

        t = (0, image_height - crop_height)
        b = (crop_height, image_height)
        l = (0, image_width - crop_width)
        r = (crop_width, image_width)
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        crop_bottom = crop_top + crop_height
        crop_right = crop_left + crop_width

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

        boxes = np.concatenate([text_boxes, global_box, region_boxes], axis=0)
        size = np.concatenate([image_size[None], image_size[None]], axis=1)
        relative_boxes = boxes / size
        return relative_boxes.astype('float32')
        
    def __getitem__(self, i):
        image_id = self.dataset_dict[i]
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

        g_tokens_type = np.ones((self.max_seq_len,), dtype=np.int64)
        ret.update({ kfg.G_TOKENS_TYPE: g_tokens_type })
        dict_as_tensor(ret)
        return ret
    
class Ensembler():
    def __init__(self, cfg):
        models = []
        num_models = len(cfg.MODEL.ENSEMBLE_WEIGHTS)
        assert num_models > 0, "cfg.MODEL.ENSEMBLE_WEIGHTS is empty"
        model = build_model(cfg)
        for i in range(num_models):
            models.append(copy.deepcopy(model))
            
            checkpointer = XmodalerCheckpointer(models[i], cfg.OUTPUT_DIR)
            checkpointer.resume_or_load(cfg.MODEL.ENSEMBLE_WEIGHTS[i], resume=False)

        self.model = Ensemble(models, cfg)

def test():
    config_file='configs/image_caption/vst/vst_eval.yaml'
    split = 'test'
    if (split == 'test'):
        output_path = 'output/result/captions_test2014_VST_results.json'
    else:
        output_path = 'output/result/captions_val2014_VST_results.json'

    cfg = get_cfg() # obtain X-modaler's default config
    tmp_cfg = cfg.load_from_file_tmp(config_file) # load custom config
    add_config(cfg, tmp_cfg) # combining default and custom configs
    cfg.merge_from_file(config_file) # load values from a file

    model = Ensembler(cfg).model

    dataset = MSCoCoTestDataset(cfg, split=split)
    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = cfg.DATALOADER.TEST_BATCH_SIZE,
            num_workers = cfg.DATALOADER.NUM_WORKERS,
            drop_last=False,
            shuffle = False,
            collate_fn=trivial_batch_collator,
        )

    model.eval()
    results = []
    with torch.no_grad():
        for data in tqdm.tqdm(data_loader):
            data = comm.unwrap_model(model).preprocess_batch(data)
            ids = data[kfg.IDS]
            
            res = model(data, use_beam_search=True, output_sents=True)

            outputs = res[kfg.OUTPUT]
            for id, output in zip(ids, outputs):
                results.append({cfg.INFERENCE.ID_KEY: int(id), cfg.INFERENCE.VALUE: output})

    with open(output_path, 'w') as fp:
        json.dump(results, fp)
        
        
if __name__ == '__main__':
    test()