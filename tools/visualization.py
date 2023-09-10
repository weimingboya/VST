import itertools
import h5py
import numpy as np
import torch
from xmodaler.config import get_cfg
from xmodaler.modeling import add_config
from xmodaler.modeling import build_model
from xmodaler.checkpoint import XmodalerCheckpointer
from xmodaler.datasets.build import build_dataset_mapper

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def test(img_id):
    config_file='configs/image_caption/vst/vst.yaml'

    cfg = get_cfg() # obtain X-modaler's default config
    tmp_cfg = cfg.load_from_file_tmp(config_file) # load custom config
    add_config(cfg, tmp_cfg) # combining default and custom configs
    cfg.merge_from_file(config_file) # load values from a file

    model = build_model(cfg)
    checkpointer = XmodalerCheckpointer(model, cfg.OUTPUT_DIR)
    cfg.MODEL.WEIGHTS = 'checkpoints/vs_transformer/xe_gpscae_mma_rl_3407/model_Epoch_00013_Iter_0147263.pth'
    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False) # load a checkpoint

    cfg.DATALOADER.SEQ_PER_SAMPLE=1
    dataset = build_dataset_mapper(cfg, name=cfg.DATASETS.TRAIN, stage="train")
    datalist = dataset.load_data(cfg)

    imgid2idx = {data['image_id']: i for i, data in enumerate(datalist)}

    input = dataset(datalist[imgid2idx[img_id]])
    input = model.preprocess_batch([input])

    model.eval()
    with torch.no_grad():
        output = model(input, use_beam_search=False, output_sents=True)

    v2t_att_map = model.encoder.layers[2].v2t_att_map
    t2v_att_map = model.encoder.layers[2].t2v_att_map
    v_cross_atts = model.decoder.g_layers[2].v2t_cross_atts
    t_cross_atts = model.decoder.g_layers[2].t2v_cross_atts
    vt_cross_atts = model.decoder.g_layers[2].vt_cross_atts

    output = output['OUTPUT']
    v_cross_atts = torch.cat(v_cross_atts,dim=0).squeeze(-2)
    t_cross_atts = torch.cat(t_cross_atts,dim=0).squeeze(-2)
    vt_cross_atts = torch.cat(vt_cross_atts,dim=0).squeeze(-2)
    return v2t_att_map[0], t2v_att_map[0], v_cross_atts, t_cross_atts, vt_cross_atts, output

def parse_boxes(image_id):
    image_size = region_f['%s_size' % image_id][()][::-1]
    image_width, image_height = image_size[0], image_size[1]
    five_ratio = 0.6
    nine_ratio = 0.4

    text_boxes = []

    # ########## orgin_boxes ###########
    # orgin_box = np.array([0,0,image_width,image_height])
    # text_boxes.append(orgin_box)

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

    region_boxes = region_f['%s_boxes' % image_id][()]
    region_boxes = region_f['%s_boxes' % image_id][()]
    region_boxes = region_boxes[:50].astype('float32')
    
    return region_boxes, text_boxes

def load_text(img_id):
    text_contents = []
    
    ctx_whole_txt = ctx[f"{img_id}/whole/texts"][:12]
    ctx_five_txt = ctx[f"{img_id}/five/texts"][:, :12]
    ctx_nine_txt = ctx[f"{img_id}/nine/texts"][:, :12]
    
    # text_contents.append(ctx_whole_txt)
    text_contents.extend(ctx_five_txt)
    text_contents.extend(ctx_nine_txt)
    return text_contents

def getEncoderResult():
    print(f'############# TEXT ({object_label}) ###############')
    v2t_att_map2 = v2t_att_map[:,object_idx+1,12:]
    text_values, text_idxs = torch.topk(v2t_att_map2, k=1, dim=-1)
    text_values = text_values.squeeze(-1).cpu().numpy().tolist()
    text_idxs = text_idxs.squeeze(-1).cpu().numpy().tolist()
    att_texts = []
    for i, idx in enumerate(text_idxs):
        group = int(idx / 12)
        group_idx = idx % 12
        att_texts.append((i, text_idxs[i], group, text_contents[group][group_idx].decode('utf-8'), text_values[i]))

    att_texts.sort(key = lambda x: x[-1], reverse=True)
    [print(i,group,text,value) for i, _, group, text, value in att_texts]
    att_text_boxes = [text_boxes[int(idx / 12)].tolist() for idx in text_idxs]

    text_idx = att_texts[0][1]
    text = text_contents[int(text_idx/12)][text_idx%12].decode('utf-8')
    print(f'############# Object ({text}) ###############')
    t2v_att_map2 = t2v_att_map[:,text_idx+12,1:]
    object_values, object_idxs = torch.topk(t2v_att_map2, k=1, dim=-1)
    object_values = object_values.squeeze(-1).cpu().numpy().tolist()
    object_idxs = object_idxs.squeeze(-1).cpu().numpy().tolist()
    att_labels = [(i, region_labels[idx].decode('utf-8'), object_values[i]) for i,idx in enumerate(object_idxs)]
    att_labels.sort(key = lambda x: x[-1], reverse=True)
    [print(i,label,value) for i, label, value in att_labels]
    att_object_boxes = [region_boxes[i].tolist() for i in object_idxs]
    
    res = {
        'att_texts': att_texts,
        'att_labels': att_labels,
        'att_text_boxes': att_text_boxes,
        'att_object_boxes': att_object_boxes,
    }
    return res

def getDecoderResult():
    words = str.split(output[0], " ")
    v_cross_atts2 = v_cross_atts[:-1,:,1:].mean(dim=1)
    t_cross_atts2 = t_cross_atts[:-1,:,12:].mean(dim=1)
    
    v_valuess, v_idxss = torch.topk(v_cross_atts2, k=3, dim=-1)
    v_valuess = v_valuess.cpu().numpy().tolist()
    v_idxss = v_idxss.cpu().numpy().tolist() #(L,3)
    
    t_valuess, t_idxss = torch.topk(t_cross_atts2, k=3, dim=-1)
    t_valuess = t_valuess.cpu().numpy().tolist()
    t_idxss = t_idxss.cpu().numpy().tolist() #(L,3)
    
    res = []
    for w, v_values, v_idxs, t_values, t_idxs in zip(words, v_valuess, v_idxss, t_valuess, t_idxss):
        v_boxes = [region_boxes[idx].tolist() for idx in v_idxs]
        v_labels = [region_labels[idx].decode('utf-8') for idx in v_idxs]
        t_boxes = [text_boxes[int(idx/12)].tolist() for idx in t_idxs]
        t_texts = [text_contents[int(idx/12)][idx%12].decode('utf-8') for idx in t_idxs]
        res.append({
            'word': w,
            'v_boxes': v_boxes,
            'v_labels': v_labels,
            'v_values': v_values,
            't_boxes': t_boxes,
            't_texts': t_texts,
            't_values': t_values,
        })
    return res

if __name__ == '__main__':    
    ctx = h5py.File('datasets/mscoco_dataset/features/txt_ctx.hdf5', 'r')
    region_f = h5py.File('datasets/mscoco_dataset/features/COCO2014_VinVL.hdf5', 'r')
    label_f = h5py.File('datasets/mscoco_dataset/features/COCO2014_VinVL_ObjectLabels.hdf5', 'r')

    img_id = '181474'

    region_boxes, text_boxes = parse_boxes(img_id)
    region_labels = label_f['%s_label' % img_id][:50]
    text_contents = load_text(img_id)

    object_idx = 1
    object_box = region_boxes[object_idx].tolist()
    object_label = region_labels[object_idx].decode('utf-8')

    v2t_att_map, t2v_att_map, v_cross_atts, t_cross_atts, vt_cross_atts, output = test(img_id)
    
    # getEncoderResult()
    res = getDecoderResult()
    [print(r) for r in res]
    pass
