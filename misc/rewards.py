from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.parameter import Parameter

import sys
sys.path.append("tools/pycider")
from pyciderevalcap.ciderD.ciderD import CiderD
import pdb

#CiderD_scorer = CiderD(df='corpus')

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

class get_self_critical_reward(nn.Module):
    def __init__(self, opt):
        super(get_self_critical_reward, self).__init__()
        self.vocab_size = opt.vocab_size
        self.st2towidx = opt.st2towidx
        self.opt = opt
        # self.st2towidx.requires_grad=False
        self.CiderD_scorer = CiderD(df=opt.cached_tokens)

    def forward(self, gen_input, greedy_input, gt_gts, ncap):

        gen_txt_seq, gen_bn_seq, gen_vis_seq = gen_input
        greedy_txt_seq, greedy_bn_seq, greedy_vis_seq = greedy_input

        self.st2towidx = self.st2towidx.type_as(gen_txt_seq)
        batch_size = gen_txt_seq.size(0)
        seq_per_img = batch_size // gt_gts.size(0)

        gen_result = gen_txt_seq.new(gen_txt_seq.size()).zero_()
        greedy_result = greedy_txt_seq.new(greedy_txt_seq.size()).zero_()

        gen_mask = gen_txt_seq < self.vocab_size
        gen_vis_seq = gen_vis_seq.view(batch_size,-1)
        gen_bn_seq = gen_bn_seq.view(batch_size, -1)

        # compose the seq
        gen_result[gen_mask] = gen_txt_seq[gen_mask]
        gen_vis_idx = gen_vis_seq[gen_mask==0]*2 + gen_bn_seq[gen_mask==0] - 1

        gen_result[gen_mask==0] = self.st2towidx[gen_vis_idx]

        greedy_mask = greedy_txt_seq < self.vocab_size
        greedy_vis_seq = greedy_vis_seq.view(batch_size,-1)
        greedy_bn_seq = greedy_bn_seq.view(batch_size, -1)

        # compose the seq
        greedy_result[greedy_mask] = greedy_txt_seq[greedy_txt_seq < self.vocab_size]
        greedy_vis_idx = greedy_vis_seq[greedy_mask==0]*2 + greedy_bn_seq[greedy_mask==0] - 1
        greedy_result[greedy_mask==0] = self.st2towidx[greedy_vis_idx]

        res = OrderedDict()
        gen_result = gen_result.cpu().numpy()
        greedy_result = greedy_result.cpu().numpy()

        for i in range(batch_size):
            res[i] = [array_to_str(gen_result[i])]
        for i in range(batch_size):
            res[batch_size + i] = [array_to_str(greedy_result[i])]

        gts = OrderedDict()
        for i in range(batch_size):
            gts_np = gt_gts[i][:ncap.data[i]].data.cpu().numpy()
            gts[i] = [array_to_str(gts_np[j]) for j in range(len(gts_np))]

        # caption = utils.decode_normal(self.opt.itow, torch.from_numpy(gen_result))
        # pdb.set_trace()
        # print(caption[0])

        # utils.decode_normal(self.opt.itow, gt_gts.data.view(-1,20))
        #_, scores = Bleu(4).compute_score(gts, res)
        #scores = np.array(scores[3])
        res = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
        gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
        _, scores = self.CiderD_scorer.compute_score(gts, res)
        # print(_)

        scores = scores[:batch_size] - scores[batch_size:]
        rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

        return rewards, _
