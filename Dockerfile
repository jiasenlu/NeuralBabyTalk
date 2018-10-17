FROM pytorch/pytorch:0.4-cuda9-cudnn7-devel

RUN apt-get update && \
    apt-get install -y \
    unzip \
    wget 

COPY . /workspace/neuralbabytalk

RUN cd /workspace/neuralbabytalk/pooling/roi_align && \
    sh make.sh

# attach COCO dataset as volume (from build context or command line argument)
ARG coco=./data/coco
VOLUME ${coco}
RUN ln -sf ${coco} /workspace/neuralbabytalk/data/coco_images

# download pretrained imagenet weights for resnet and vgg
RUN mkdir /workspace/neuralbabytalk/data/pretrained_models && \
    cd /workspace/neuralbabytalk/data/pretrained_models && \
    wget https://www.dropbox.com/sh/67fc8n6ddo3qp47/AADUMRqlcvjv4zqBX6K2L8c2a && \
    unzip AADUMRqlcvjv4zqBX6K2L8c2a && rm AADUMRqlcvjv4zqBX6K2L8c2a
