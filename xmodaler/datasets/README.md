# Use Builtin Datasets

A dataset can be used by wrapping it into a torch Dataset. This document explains how to setup the builtin datasets so they can be used by X-modaler.

X-modaler has builtin support for a few datasets (e.g., MSCOCO or MSVD). The corresponding dataset wrappers are provided in `./xmodaler/datasets`:
```
xmodaler/datasets/
  images/
    mscoco.py
  videos/
    msvd.py  
```
You can specify which dataset wrapper to use by `DATASETS.TRAIN`, `DATASETS.VAL` and `DATASETS.TEST` in the config file. 

# Expected structure for xmodaler
First, download the [dataset files](https://drive.google.com/drive/folders/1vx9n7tAIt8su0y_3tsPJGvMPBMm8JLCZ?usp=sharing), [pre-trained models](https://drive.google.com/drive/folders/14N0MHJl0MvzuXa6RAmauiHfvFmaAZ0Xn?usp=sharing) and [coco_caption](https://github.com/ruotianluo/coco-caption).

```
xmodaler
coco_caption
datasets/
  mscoco_dataset
  msvd_dataset
  msrvtt_dataset
  ConceptualCaptions
  VQA
  VCR
  flickr30k
pretrain/
  BERT
  TDEN
  Uniter
```

## Expected dataset structure for [COCO](https://cocodataset.org/#download):

```
mscoco_dataset/
  mscoco_caption_anno_train.pkl
  mscoco_caption_anno_val.pkl
  mscoco_caption_anno_test.pkl
  vocabulary.txt
  captions_val5k.json
  captions_test5k.json
  # image files that are mentioned in the corresponding json
features/
  up_down/
      *.npz
```

## Expected dataset structure for [MSVD](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/):

```
msvd_dataset/
  msvd_caption_anno_train.pkl
  msvd_caption_anno_val.pkl
  msvd_caption_anno_test.pkl
  vocabulary.txt
  captions_val.json
  captions_test.json
  # videos files that are mentioned in the corresponding json
features/
  resnet152/
    *.npy
```

## Expected dataset structure for [MSR-VTT](http://ms-multimedia-challenge.com/2017/dataset):

```
msrvtt_dataset/
  msrvtt_caption_anno_train.pkl
  msrvtt_caption_anno_val.pkl
  msrvtt_caption_anno_test.pkl
  vocabulary.txt
  captions_val.json
  captions_test.json
  # videos files that are mentioned in the corresponding json
msrvtt_torch/
  feature/
    resnet152/
      *.npy
```

When the dataset wrapper and data files are ready, you need to specify the corresponding paths to these data files in the config file. For example, 
```
DATALOADER:
	FEATS_FOLDER: 'datasets/mscoco_dataset/features/up_down'    # feature folder
	ANNO_FOLDER: 'datasets/mscoco_dataset' # annotation folders
INFERENCE:
	VOCAB: 'datasets/mscoco_dataset/vocabulary.txt'
	VAL_ANNFILE: 'datasets/mscoco_dataset/captions_val5k.json'
	TEST_ANNFILE:  'datasets/mscoco_dataset/captions_test5k.json'
```
