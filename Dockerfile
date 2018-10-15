FROM pytorch/pytorch:0.4-cuda9-cudnn7-devel

RUN apt-get update && \
    apt-get install -y wget

COPY . /workspace/neuralbabytalk

RUN cd /workspace/neuralbabytalk/pooling/roi_align && \
    sh make.sh

# attach COCO dataset as volume (from build context or command line argument)
ARG coco=./data/coco
VOLUME ${coco}
RUN ln -sf ${coco} /workspace/neuralbabytalk/data/coco_images
