# Neural Baby Talk

![teaser results](demo/img1.png)

## requirement

Inference:

- [pytorch](http://pytorch.org/)
- [torchvision](https://github.com/pytorch/vision)
- [torchtext](https://github.com/pytorch/text)

Data Preparation:

- [stanford-corenlp-wrapper](https://github.com/Lynten/stanford-corenlp)
- [stanford-corenlp](https://stanfordnlp.github.io/CoreNLP/)

Evaluation:

- [coco-caption](https://github.com/jiasenlu/coco-caption): Download the modified version of coco-caption and put it under `tools/`


## Demo


## Training and Evaluation
### Data Preparation
Head to `data/README.md`, and prepare the data for training and evaluation.

### Standard Image Captioning
##### Training (COCO)

First, modify the cofig file `cfgs/normal_coco_res101.yml` with the correct file path.

```python
python main.py --path_opt cfgs/normal_coco_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30
```
##### Evaluation (COCO)
Download Pre-trained model from [Link](https://www.dropbox.com/s/6buajkxm9oed1jp/coco_nbt_1024.tar.gz?dl=0). Extract the tar.zip file and put it under `save/`.

```python
python main.py --path_opt cfgs/normal_coco_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --inference_only True --beam_size 3 --start_from save/coco_nbt_1024
```

##### Training (Flickr30k)
Modify the cofig file `cfgs/normal_flickr_res101.yml` with the correct file path.

```python
python main.py --path_opt cfgs/normal_flickr_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30
```

##### Evaluation (Flickr30k)
Download Pre-trained model from [Link](https://www.dropbox.com/s/6buajkxm9oed1jp/coco_nbt_1024.tar.gz?dl=0). Extract the tar.zip file and put it under `save/`.

```python
python main.py --path_opt cfgs/normal_coco_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --inference_only True --beam_size 3 --start_from save/coco_nbt_1024
```

### Robust Image Captioning

##### Training

### Novel Object Captioning

##### Training

## More Visualization Results
![teaser results](demo/img2.png)

## Reference
If you use this code as part of any published research, please acknowledge the following paper

```
@misc{Lu2018Neural,
author = {Lu, Jiasen and Yang, Jianwei and Batra, Dhruv and Parikh, Devi},
title = {Neural Baby Talk},
journal = {CVPR},
year = {2018}
}
```
