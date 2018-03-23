## Data Preparation for Neural Baby Talk
### Image Dataset

- COCO: Download coco images from [link](http://cocodataset.org/#download), we need `2014 training` images and `2014 val` images. You should put the image in some directory, denoted as `$IMAGE_ROOT`.

- Flickr30k: Download flickr30k entity images from [link](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/), you may need to fill a form to get the annotations.

### Pretrained CNN weight
- Download pretrained CNN weight from [link](https://www.dropbox.com/sh/67fc8n6ddo3qp47/AADUMRqlcvjv4zqBX6K2L8c2a?dl=0), and put it into `/data`

### COCO
- Download the preprocessed Karpathy's split of coco caption from [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). Extract `dataset_coco.json` from the zip file and copy it into `coco/`.
- Download COCO 2014 Train/Val annotations from [link](http://images.cocodataset.org/annotations/annotations_trainval2014.zip). Extract the zip file and put the json file under `coco/annotations/`
- Download stanford core nlp tools and modified the `scripts/prepro_dic_coco.py` with correct stanford core nlp location. (In my experiment, I use the the version of `stanford-corenlp-full-2017-06-09` [link](https://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip))
- You can either download the preprocessed data from [here](https://www.dropbox.com/s/1t9nrbevzqn93to/coco.tar.gz?dl=0) or you can use the pre-process script to generate the data. Under the `root` directory, run the following command to pre-process the data.
```
python prepro/prepro_dic_coco.py --input_json data/coco/dataset_coco.json --split normal --output_dic_json data/coco/dic_coco.json --output_cap_json data/coco/cap_coco.json
```
- Download the pre-extracted coco detection result from [link](https://www.dropbox.com/s/2gzo4ops5gbjx5h/coco_detection.h5.tar.gz?dl=0) and extract the tar.gz file and copy it into `coco/`. You can also extract using our reimplementation of faster rcnn code, or any exsiting detection framework. The format of bounding box data will added later.
- After all these steps, we are ready to train the model for coco :)

### Flickr30k
- Download the preprocessed Karpathy's split of coco caption from [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). Extract `dataset_flickr30k.json` from the zip file and copy it into `flickr30k/`.
- Download the preprocessed Flickr30k annotation for NeuralBabyTalk (annotations that linking the nouns to specific bounding box) from [link](https://www.dropbox.com/s/h4ru86ocb10axa1/flickr30k_cleaned_class.json.tar.gz?dl=0). Extract the tar.gz file and copy it into `flickr30k/`.
- Download the stanford corenlp as COCO's instruction. 
- You can either download the preprocessed data from [here](https://www.dropbox.com/s/twve5exs8qj9xgd/flickr30k.tar.gz?dl=0) or you can use the pre-process script to generate the data. Under the `root` directory, run the following command to pre-process the data.
```
python prepro/prepro_dic_flickr.py --input_json data/flickr30k/dataset_flickr30k.json --input_class_name data/flickr30k/flickr30k_class_name.txt
```
- Download the pre-extracted flickr30k detection result from [link](https://www.dropbox.com/s/5o6so7h4xq5ki1t/flickr30k_detection.h5.tar.gz?dl=0) and extract the tar.gz file and copy it into `flickr30k/`. You can also extract using our reimplementation of faster rcnn code, or any exsiting detection framework. The format of bounding box data will added later.
- After all these steps, we are ready to train the model for flickr30k :)

### Robust-COCO
- Follow the instructions as COCO (1-3, 5).
- You can either download the preprocessed data from [here](https://www.dropbox.com/s/tevyub9rxz6d22l/coco_robust.tar.gz?dl=0) or you can use the pre-process script to generate the data. Under the `root` directory, run the following command to pre-process the data.
```
python prepro/prepro_dic_coco.py --input_json data/coco/dataset_coco.json --split robust --output_dic_json data/robust_coco/dic_coco.json --output_cap_json data/robust_coco/cap_coco.json
```

### NOC-COCO
- Follow the instructions as COCO (1-3). 
- You can either download the preprocessed data from [here](https://www.dropbox.com/s/tevyub9rxz6d22l/coco_robust.tar.gz?dl=0) or you can use the pre-process script to generate the data. Under the `root` directory, run the following command to pre-process the data.
```
python prepro/prepro_dic_coco.py --input_json data/coco/dataset_coco.json --split noc --output_dic_json data/noc_coco/dic_coco.json --output_cap_json data/noc_coco/cap_coco.json
```
- Download the pre-extracted coco detection result trained on `train2014` from [link](https://www.dropbox.com/s/2gzo4ops5gbjx5h/coco_detection.h5.tar.gz?dl=0) and extract the tar.gz file and copy it into `coco/`. You can also extract using our reimplementation of faster rcnn code, or any exsiting detection framework. The format of bounding box data will added later.

