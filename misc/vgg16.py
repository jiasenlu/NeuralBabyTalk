# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import pdb
import torchvision.models as models

class vgg16(nn.Module):
  def __init__(self, opt, pretrained=True):
    super(vgg16, self).__init__()

    self.model_path = '%s/imagenet_weights/vgg16_caffe.pth' %(opt.data_path)
    self.pretrained = pretrained

    vgg = models.vgg16()
    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
    self.fc = vgg.classifier
    self.pooling = nn.AdaptiveAvgPool2d((7,7))
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    # not using the last maxpool layer
    self.cnn_net = nn.Sequential(*list(vgg.features._modules.values())[:-1])

  def forward(self, img):

    conv_feat = self.cnn_net(img)
    pooled_conv_feat = self.pooling(conv_feat)

    pooled_conv_feat_flat = pooled_conv_feat.view(pooled_conv_feat.size(0), -1)
    fc_feat = self.fc(pooled_conv_feat_flat)

    return conv_feat, fc_feat