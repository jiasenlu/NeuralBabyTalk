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

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq, vocab_size):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix == 0 or ix == vocab_size:
                break
            else:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]

        out.append(txt)
    return out

def repackage_hidden(h, batch_size):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data.resize_(h.size(0), batch_size, h.size(2)).zero_())
    else:
        return tuple(repackage_hidden(v, batch_size) for v in h)

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.view(-1)
        reward = reward.view(-1)
        mask = (seq>0).float().view(-1)
        output = - input * reward * Variable(mask)
        output = torch.sum(output) / torch.sum(mask)
        return output

class LMCriterion(nn.Module):
    def __init__(self):
        super(LMCriterion, self).__init__()

    def forward(self, input, target):
        # truncate to the same size

        mask = target.data.gt(0)  # generate the mask
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).view(-1,1)
        if isinstance(input, Variable):
            mask = Variable(mask, volatile=input.volatile)

        target = target.contiguous().view(-1,1)
        logprob_select = torch.gather(input.view(-1, input.size(2)), 1, target)
        out = torch.masked_select(logprob_select, mask)

        loss = -torch.sum(out) # get the average loss.
        loss = loss / torch.sum(target.data.gt(0))

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

def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/captions_val2014.json'
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
    cocoEval = COCOEvalCap(coco, cocoRes)
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


def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key] 
    return dict_to