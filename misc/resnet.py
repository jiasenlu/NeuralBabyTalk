from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class resnet(nn.Module):
  def __init__(self, _num_layers=101, _fixed_block=1, _att_size=14, pretrained=True):
    super(resnet, self).__init__()
    self._num_layers = _num_layers
    self._fixed_block = _fixed_block
    self._att_size = _att_size
    if self._num_layers == 50:
      self.resnet = models.resnet50(pretrained=pretrained)

    elif self._num_layers == 101:
      self.resnet = models.resnet101(pretrained=pretrained)

    elif self._num_layers == 152:
      self.resnet = models.resnet152(pretrained=pretrained)
    else:
      raise NotImplementedError

    # Fix blocks
    for p in self.resnet.bn1.parameters(): p.requires_grad=False
    for p in self.resnet.conv1.parameters(): p.requires_grad=False
    assert (0 <= _fixed_block <= 4)
    if _fixed_block >= 4:
      for p in self.resnet.layer4.parameters(): p.requires_grad=False
    if _fixed_block >= 3:
      for p in self.resnet.layer3.parameters(): p.requires_grad=False
    if _fixed_block >= 2:
      for p in self.resnet.layer2.parameters(): p.requires_grad=False
    if _fixed_block >= 1:
      for p in self.resnet.layer1.parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.resnet.apply(set_bn_fix)
    
    self.cnn_net = nn.Sequential(self.resnet.conv1, self.resnet.bn1,self.resnet.relu, 
      self.resnet.maxpool,self.resnet.layer1,self.resnet.layer2,self.resnet.layer3, self.resnet.layer4)

  def forward(self, img):
    conv_feat = self.cnn_net(img)
    fc_feat = conv_feat.mean(3).mean(2)
    conv_feat = F.adaptive_avg_pool2d(conv_feat,[self._att_size, self._att_size])

    return conv_feat, fc_feat
  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.resnet.eval()
      if self._fixed_block <= 3:
        self.resnet.layer4.train()
      if self._fixed_block <= 2:
        self.resnet.layer3.train()
      if self._fixed_block <= 1:
        self.resnet.layer2.train()
      if self._fixed_block <= 0:
        self.resnet.layer1.train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.resnet.apply(set_bn_eval)