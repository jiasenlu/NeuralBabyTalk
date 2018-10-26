FROM pytorch/pytorch:0.4-cuda9-cudnn7-devel

# ----------------------------------------------------------------------------
# -- install apt and pip dependencies
# ----------------------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y \
    ant \
    ca-certificates-java \
    nano \
    openjdk-8-jdk \
    python2.7 \
    unzip \
    wget && \
    apt-get clean

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN update-ca-certificates -f && export JAVA_HOME

RUN pip install Cython && pip install h5py \
    matplotlib \
    nltk \
    numpy \
    pycocotools \
    scikit-image \
    stanfordcorenlp \
    torchtext \
    tqdm && python -c "import nltk; nltk.download('punkt')"

# ----------------------------------------------------------------------------
# -- copy repository and build roi_align shared object
# ----------------------------------------------------------------------------

COPY . /workspace/neuralbabytalk
RUN git clone https://github.com/jwyang/faster-rcnn.pytorch /workspace/faster-rcnn.pytorch && \
    cd /workspace/faster-rcnn.pytorch/lib && sh make.sh && \
    cp /workspace/faster-rcnn.pytorch/lib/model/roi_align/_ext/roi_align/_roi_align.so \
       /workspace/neuralbabytalk/pooling/roi_align/_ext/roi_align


# ----------------------------------------------------------------------------
# -- download pretrained imagenet weights for resnet-101
# ----------------------------------------------------------------------------

RUN mkdir /workspace/neuralbabytalk/data/imagenet_weights && \
    cd /workspace/neuralbabytalk/data/imagenet_weights && \
    wget --quiet https://www.dropbox.com/sh/67fc8n6ddo3qp47/AAACkO4QntI0RPvYic5voWHFa/resnet101.pth


# ----------------------------------------------------------------------------
# -- download Karpathy's preprocessed captions datasets and corenlp jar
# ----------------------------------------------------------------------------

RUN cd /workspace/neuralbabytalk/data && \
    wget --quiet http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip && \
    unzip caption_datasets.zip && \
    mv dataset_coco.json coco/ && \
    mv dataset_flickr30k.json flickr30k/ && \
    rm caption_datasets.zip dataset_flickr8k.json

RUN cd /workspace/neuralbabytalk/prepro && \
    wget --quiet https://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip && \
    unzip stanford-corenlp-full-2017-06-09.zip && \
    rm stanford-corenlp-full-2017-06-09.zip


# ----------------------------------------------------------------------------
# -- download preprocessed COCO detection output HDF file and pretrained model
# ----------------------------------------------------------------------------

RUN cd /workspace/neuralbabytalk/data/coco && \
    wget --quiet https://www.dropbox.com/s/2gzo4ops5gbjx5h/coco_detection.h5.tar.gz && \
    tar -xzvf coco_detection.h5.tar.gz && \
    rm coco_detection.h5.tar.gz

RUN mkdir -p /workspace/neuralbabytalk/save && \
    cd /workspace/neuralbabytalk/save && \
    wget --quiet https://www.dropbox.com/s/6buajkxm9oed1jp/coco_nbt_1024.tar.gz && \
    tar -xzvf coco_nbt_1024.tar.gz && \
    rm coco_nbt_1024.tar.gz

WORKDIR /workspace/neuralbabytalk
RUN python prepro/prepro_dic_coco.py \
    --input_json data/coco/dataset_coco.json \
    --split normal \
    --output_dic_json data/coco/dic_coco.json \
    --output_cap_json data/coco/cap_coco.json
EXPOSE 8888
