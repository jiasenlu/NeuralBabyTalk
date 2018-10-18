FROM pytorch/pytorch:0.4-cuda9-cudnn7-devel

RUN apt-get update && \
    apt-get install -y \
    unzip \
    wget 


# ----------------------------------------------------------------------------
# -- copy repository and build roi_align shared object
# ----------------------------------------------------------------------------

COPY . /workspace/neuralbabytalk
RUN cd /workspace/neuralbabytalk/pooling/roi_align && \
    sh make.sh

# ----------------------------------------------------------------------------
# -- attach COCO dataset as volume
# ----------------------------------------------------------------------------

# 'coco_images' should have sub-directories 'train2014' and 'val2014'
ARG coco_images=./data/coco/images
ARG coco_anns=./data/coco/annotations

# 'coco_anns' should have 'captions_train2014.json' and 'captions_val2014.json'
VOLUME ${coco_images}
VOLUME ${coco_anns}
RUN ln -sf ${coco_images} /workspace/neuralbabytalk/data/coco/images && \
    ln -sf ${coco_anns} /workspace/neuralbabytalk/data/coco/annotations


# ----------------------------------------------------------------------------
# -- download pretrained imagenet weights for resnet-101
# ----------------------------------------------------------------------------

RUN mkdir /workspace/neuralbabytalk/data/pretrained_models && \
    cd /workspace/neuralbabytalk/data/pretrained_models && \
    wget https://www.dropbox.com/sh/67fc8n6ddo3qp47/AAACkO4QntI0RPvYic5voWHFa/resnet101.pth


# ----------------------------------------------------------------------------
# -- download Karpathy's preprocessed captions datasets
# ----------------------------------------------------------------------------

RUN cd /workspace/neuralbabytalk/data && \
    wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip && \
    unzip caption_datasets.zip && \
    mv dataset_coco.json coco/ && \
    mv dataset_flickr30k.json flickr30k/ && \
    rm caption_datasets.zip dataset_flickr8k.json
