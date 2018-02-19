from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb
import os
import json
from misc.bbox_transform import bbox_overlaps_batch
import numbers
import random
import math
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import types
import warnings
import torch.nn.functional as F
import sys
sys.path.append("tools/sentence_gen_tools")
from coco_eval import score_dcc
import matplotlib.pyplot as plt
import matplotlib.patches as patches

noc_object = ['bus', 'bottle', 'couch', 'microwave', 'pizza', 'racket', 'suitcase', 'zebra']

noc_word_map = {'bus':'car', 'bottle':'cup',
                'couch':'chair', 'microwave':'oven',
                'pizza': 'cake', 'tennis racket': 'baseball bat',
                'suitcase': 'handbag', 'zebra': 'horse'}

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(itow, itod, ltow, itoc, wtod, seq, bn_seq, fg_seq, vocab_size, opt):
    N, D = seq.size()

    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            if j >= 1:
                txt = txt + ' '
            ix = seq[i,j]
            if ix > vocab_size:
                det_word = itod[fg_seq[i,j]]
                det_class = itoc[wtod[det_word]]
                if opt.decode_noc and det_class in noc_object:
                    det_word = det_class

                if (bn_seq[i,j] == 1) and det_word in ltow:
                    word = ltow[det_word]
                else:
                    word = det_word
                # word = '[ ' + word + ' ]'
            else:
                if ix == 0:
                    break
                else:
                    word = itow[str(ix)]
            txt = txt + word
        out.append(txt)
    return out

def decode_sequence_det(itow, itod, ltow, itoc, wtod, seq, bn_seq, fg_seq, vocab_size, opt):
    N, D = seq.size()
    det_idxs = []
    det_words = []
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            if j >= 1:
                txt = txt + ' '
            ix = seq[i,j]
            if ix > vocab_size:
                det_word = itod[fg_seq[i,j]]
                det_class = itoc[wtod[det_word]]
                if opt.decode_noc and det_class in noc_object:
                    det_word = det_class

                if (bn_seq[i,j] == 1) and det_word in ltow:
                    word = ltow[det_word]
                else:
                    word = det_word
                # word = '[ ' + word + ' ]'
                det_words.append(word)
                idx = ix - vocab_size - 1
                det_idxs.append(idx)
            else:
                if ix == 0:
                    break
                else:
                    word = itow[str(ix)]
            txt = txt + word
        out.append(txt)
    return out, det_idxs, det_words


def decode_normal(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out

def decode_sequence_bbox(itow, itod, ltow, seq, bn_seq, fg_seq, vocab_size, opt):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            if j >= 1:
                txt = txt + ' '
            ix = seq[i,j]
            if ix > vocab_size:
                det_word = itod[fg_seq[i,j]]
                if (bn_seq[i,j] == 1) and det_word in ltow:
                    word = ltow[det_word]
                else:
                    word = det_word
                # word = '[ ' + word + ' ]'
            else:
                if ix == 0:
                    break
                else:
                    word = itow[str(ix)]
            txt = txt + word
        out.append(txt)


def repackage_hidden(h, batch_size):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data.resize_(h.size(0), batch_size, h.size(2)).zero_())
    else:
        return tuple(repackage_hidden(v, batch_size) for v in h)

# class RewardCriterion(nn.Module):
#     def __init__(self):
#         super(RewardCriterion, self).__init__()
#
#     def forward(self, input, seq, reward):
#         input = input.view(-1)
#         reward = reward.view(-1)
#         mask = (seq>0).float().view(-1)
#         output = - input * reward * Variable(mask)
#         output = torch.sum(output) / torch.sum(mask)
#
#         # print(output)
#         return output

class RewardCriterion(nn.Module):
    def __init__(self, opt):
        super(RewardCriterion, self).__init__()
        self.vocab_size = opt.vocab_size

    def forward(self, seq, bn_seq, fg_seq, seqLogprobs, bnLogprobs, fgLogprobs, reward):

        seqLogprobs = seqLogprobs.view(-1)
        reward = reward.view(-1)
        fg_seq = fg_seq.squeeze()
        seq_mask = torch.cat((seq.new(seq.size(0),1).fill_(1).byte(), seq.gt(0)[:,:-1]),1).view(-1) #& fg_seq.eq(0)).view(-1)
        seq_mask = Variable(seq_mask)
        seq_out = - torch.masked_select(seqLogprobs * reward, seq_mask)
        seq_out = torch.sum(seq_out) / torch.sum(seq_mask.data)

        bnLogprobs = bnLogprobs.view(-1)
        bn_mask = fg_seq.gt(0).view(-1)
        bn_mask = Variable(bn_mask)

        bn_out = - torch.masked_select(bnLogprobs * reward, bn_mask)
        bn_out = torch.sum(bn_out) / max(torch.sum(bn_mask.data),1)

        fgLogprobs = fgLogprobs.view(-1)
        fg_out = - torch.masked_select(fgLogprobs * reward, bn_mask)
        fg_out = torch.sum(fg_out) / max(torch.sum(bn_mask.data),1)

        return seq_out, bn_out, fg_out

class LMCriterion(nn.Module):
    def __init__(self, opt):
        super(LMCriterion, self).__init__()
        self.vocab_size = opt.vocab_size

    def forward(self, txt_input, vis_input, target):
        target_copy = target.clone()

        vis_mask = Variable((target.data > self.vocab_size)).view(-1,1)
        txt_mask = target.data.gt(0)  # generate the mask
        txt_mask = torch.cat([txt_mask.new(txt_mask.size(0), 1).fill_(1), txt_mask[:, :-1]], 1)
        txt_mask[target.data > self.vocab_size] = 0

        vis_out = - torch.masked_select(vis_input, vis_mask)
        target.data[target.data > self.vocab_size]=0
        # truncate to the same size

        target = target.view(-1,1)
        txt_select = torch.gather(txt_input, 1, target)
        if isinstance(txt_input, Variable):
            txt_mask = Variable(txt_mask)
        txt_out = - torch.masked_select(txt_select, txt_mask.view(-1,1))

        loss = (torch.sum(txt_out)+torch.sum(vis_out)) / (torch.sum(txt_mask.data) + torch.sum(vis_mask.data))

        return loss

class BNCriterion(nn.Module):
    def __init__(self, opt):
        super(BNCriterion, self).__init__()

    def forward(self, input, target):

        target = target.view(-1,1)-1
        bn_mask = target.data.ne(-1)  # generate the mask
        if isinstance(input, Variable):
            bn_mask = Variable(bn_mask)

        if torch.sum(bn_mask.data) > 0:
            new_target = target.data.clone()
            new_target[new_target<0] = 0
            select = torch.gather(input.view(-1,2), 1, Variable(new_target))

            out = - torch.masked_select(select, bn_mask)
            loss = torch.sum(out) / torch.sum(bn_mask.data)
        else:
            loss = Variable(input.data.new(1).zero_()).float()

        return loss

class FGCriterion(nn.Module):
    def __init__(self, opt):
        super(FGCriterion, self).__init__()

    def forward(self, input, target):

        target = target.view(-1,1)
        input = input.view(-1, input.size(2))

        select = torch.gather(input, 1, target)

        attr_mask = target.data.gt(0)  # generate the mask
        if isinstance(input, Variable):
            attr_mask = Variable(attr_mask)

        if torch.sum(attr_mask.data) > 0:
            out = - torch.masked_select(select, attr_mask)
            loss = torch.sum(out) / torch.sum(attr_mask.data)
        else:
            loss = Variable(input.data.new(1).zero_()).float()

        return loss


def set_lr(optimizer, decay_factor):
    for group in optimizer.param_groups:
        group['lr'] = group['lr'] * decay_factor

# def clip_gradient(optimizer, grad_clip):
#     totalnorm = 0
#     for p in model.parameters():
#         if p.requires_grad:
#             if p.grad is not None:
#                 modulenorm = p.grad.data.norm()
#                 totalnorm += modulenorm ** 2
#     totalnorm = np.sqrt(totalnorm)

#     norm = clip_norm / max(totalnorm, clip_norm)
#     for p in model.parameters():
#         if p.requires_grad:
#             if p.grad is not None:
#                 p.grad.mul_(norm)

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def noc_eval(pred, model_id, split, opt):
    # first, we need to arrange the generated caption based on the novel object class.
    # construct based on the class.
    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', 'noc_' + model_id + '_' + split + '.json')

    gt_template_novel= '%s/coco_noc/annotations/'%(opt.data_path) + 'captions_split_set_%s_val_%s_novel2014.json'
    gt_template_train= '%s/coco_noc/annotations/'%(opt.data_path) + 'captions_split_set_%s_val_%s_train2014.json' 
    out = score_dcc(gt_template_novel, gt_template_train, pred, noc_object, split, cache_path)

    return out

def language_eval(dataset, preds, model_id, split, opt):
    import sys
    sys.path.append("tools/coco-caption")
    if dataset == 'coco':
        annFile = 'tools/coco-caption/annotations/captions_val2014.json'
    elif dataset == 'flickr30k':
        annFile = 'tools/coco-caption/annotations/caption_flickr30k.json'

    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')

    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes, opt.cider_df)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score
    imgToEval = cocoEval.imgToEval

    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption

    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))

def pad(img, padding, fill=0):
    """Pad the given PIL Image on all sides with the given "pad" value.
    Args:
        img (PIL Image): Image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
    Returns:
        PIL Image: Padded image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')

    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    return ImageOps.expand(img, border=padding, fill=fill)

class RandomCropWithBbox(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, proposals, bboxs):
        """
        Args:
            img (PIL Image): Image to be cropped.
            proposals, bboxs: proposals and bboxs to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = pad(img, self.padding)

        i, j, h, w = self.get_params(img, self.size)

        proposals[:,1] = proposals[:,1] - i
        proposals[:,3] = proposals[:,3] - i
        proposals[:, 1] = np.clip(proposals[:, 1], 0, h - 1)
        proposals[:, 3] = np.clip(proposals[:, 3], 0, h - 1)

        proposals[:,0] = proposals[:,0] - j
        proposals[:,2] = proposals[:,2] - j
        proposals[:, 0] = np.clip(proposals[:, 0], 0, w - 1)
        proposals[:, 2] = np.clip(proposals[:, 2], 0, w - 1)

        bboxs[:,1] = bboxs[:,1] - i
        bboxs[:,3] = bboxs[:,3] - i
        bboxs[:, 1] = np.clip(bboxs[:, 1], 0, h - 1)
        bboxs[:, 3] = np.clip(bboxs[:, 3], 0, h - 1)

        bboxs[:,0] = bboxs[:,0] - j
        bboxs[:,2] = bboxs[:,2] - j
        bboxs[:, 0] = np.clip(bboxs[:, 0], 0, w - 1)
        bboxs[:, 2] = np.clip(bboxs[:, 2], 0, w - 1)

        return crop(img, i, j, h, w), proposals, bboxs

def resize_bbox(bbox, width, height, rwidth, rheight):
    """
    resize the bbox from height width to rheight rwidth
    bbox: x,y,width, height.
    """
    width_ratio = rwidth / float(width)
    height_ratio = rheight / float(height)

    bbox[:,0] = bbox[:,0] * width_ratio
    bbox[:,2] = bbox[:,2] * width_ratio
    bbox[:,1] = bbox[:,1] * height_ratio
    bbox[:,3] = bbox[:,3] * height_ratio

    return bbox

def bbox_overlaps(rois, gt_box):
    max_bbox = gt_box.size(1)
    batch_size = rois.size(0)
    max_rois = rois.size(1)

    overlaps = bbox_overlaps_batch(rois[:,:,:4], gt_box)
    overlaps = overlaps.view(batch_size, 1, max_rois, max_bbox)\
                .expand(batch_size, 5, max_rois, max_bbox).contiguous()\
                .view(-1, max_rois, max_bbox)

    return overlaps


def bbox_target(rois, mask, overlaps, seq, seq_update, vocab_size):

    rois = rois.data
    mask = mask.data.contiguous()
    gt_labels = seq.data[:,0].clone()
    overlaps_copy = overlaps.clone()

    gt_labels = gt_labels-vocab_size
    gt_labels[gt_labels<0] = 0
    max_rois = rois.size(1)
    batch_size = rois.size(0)

    overlaps_copy.masked_fill_(mask.view(batch_size*5, 1, -1).expand_as(overlaps_copy), 0)
    max_overlaps, gt_assignment = torch.max(overlaps_copy, 2)
    # get the scores.
    scores = rois[:,:,5].contiguous().view(batch_size,1,-1).expand(batch_size,5,max_rois)\
                .contiguous().view(-1, max_rois)
    roi_labels = rois[:,:,4].contiguous().view(batch_size,1,-1).expand(batch_size,5,max_rois)\
                .contiguous().view(-1, max_rois).long()

    gt_labels = gt_labels.view(-1,1).expand_as(roi_labels)

    # get the labels.
    labels = ((max_overlaps > 0.5) & (scores >= 0.5) & (roi_labels == gt_labels)).float()
    no_proposal_idx = (labels.sum(1) > 0) != (seq.data[:,2] > 0)

    # seq_new = seq.clone()
    if no_proposal_idx.sum() > 0:
        seq_update[:,0][no_proposal_idx] = seq_update[:,3][no_proposal_idx]
        seq_update[:,1][no_proposal_idx] = 0
        seq_update[:,2][no_proposal_idx] = 0

    return labels#, seq_new

def _affine_grid_gen(rois, input_size, grid_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))

    return grid


def _jitter_boxes(gt_boxes, jitter=0.05):
    """
    """
    jittered_boxes = gt_boxes.copy()
    ws = jittered_boxes[:, 2] - jittered_boxes[:, 0] + 1.0
    hs = jittered_boxes[:, 3] - jittered_boxes[:, 1] + 1.0
    width_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * ws
    height_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * hs
    jittered_boxes[:, 0] += width_offset
    jittered_boxes[:, 2] += width_offset
    jittered_boxes[:, 1] += height_offset
    jittered_boxes[:, 3] += height_offset

    return jittered_boxes


def vis_detections(ax, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    bbox = tuple(int(np.round(x)) for x in dets[:4])
    score = dets[-1]

    ax.add_patch(
        patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2]-bbox[0],
            bbox[3]-bbox[1],
            fill=False,      # remove background
            lw=3,
            color='green'
        )
    )
    ax.text(bbox[0], bbox[1] + 15, '%s: %.3f' % (class_name, score),
            color='red', fontsize=10)
    return ax

import operator as op


import itertools
def cbs_beam_tag(num):
    tags = []
    for i in range(num+1):
        for tag in itertools.combinations(range(num), i):
            tags.append(tag)        
    return len(tags), tags

def cmpSet(t1, t2):
    return sorted(t1) == sorted(t2)

def containSet(List1, t1):
    # List1: return the index that contain
    # t1: tupple we want to match

    if t1 == tuple([]):
        return [tag for tag in List1 if len(tag) <= 1]
    else:
        List = []
        for t in List1:
            flag = True
            for tag in t1:
                if tag not in t:
                    flag = False

            if flag == True and len(t) <= len(t1)+1:
                List.append(t)
    return List
