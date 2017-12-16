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

import sys
sys.path.append("pycider")
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
    def __init__(self):
        super(get_self_critical_reward, self).__init__()
        self.CiderD_scorer = CiderD(df='coco-train-idxs')

    def forward(self, gen_input, greedy_input, gt_gts, ncap):

        batch_size = gen_input.size(0)
        seq_per_img = batch_size // gt_gts.size(0)
        
        res = OrderedDict()
        gen_result = gen_input.cpu().numpy()
        greedy_result = greedy_input.cpu().numpy()

        for i in range(batch_size):
            res[i] = [array_to_str(gen_result[i])]
        for i in range(batch_size):
            res[batch_size + i] = [array_to_str(greedy_result[i])]

        gts = OrderedDict()
        for i in range(batch_size):
            gts_np = gt_gts[i][:ncap.data[i]].data.cpu().numpy()
            gts[i] = [array_to_str(gts_np[j]) for j in range(len(gts_np))]

        #_, scores = Bleu(4).compute_score(gts, res)
        #scores = np.array(scores[3])
        res = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
        gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
        _, scores = self.CiderD_scorer.compute_score(gts, res)
        # print('Cider scores:', _)

        scores = scores[:batch_size] - scores[batch_size:]

        rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

        return rewards, _
