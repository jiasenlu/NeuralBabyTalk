from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
from misc.resnet import resnet
from misc.vgg16 import vgg16
from misc.CaptionModel import CaptionModel
from misc.rewards import get_self_critical_reward
from pooling.roi_align.modules.roi_align import RoIAlignAvg
from pooling.roi_crop.modules.roi_crop import _RoICrop
# from roi_crop.modules.gridgen import _AffineGridGen
from misc.utils import _affine_grid_gen
from torch.autograd import Variable
import math

import numpy as np
# from misc.adaptiveLSTMCell import adaptiveLSTMCell
import pdb

class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.image_crop_size = opt.image_crop_size
        self.vocab_size = opt.vocab_size
        self.detect_size = opt.detect_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.finetune_cnn = opt.finetune_cnn
        self.cbs = opt.cbs
        self.cbs_mode = opt.cbs_mode
        self.seq_per_img = 5
        if opt.cnn_backend == 'vgg16':
            self.stride = 16
        else:
            self.stride = 32

        self.att_size = int(opt.image_crop_size / self.stride)
        self.tiny_value = 1e-8

        self.pool_feat_size = self.att_feat_size + 300 * 2
        self.ss_prob = 0.0   # Schedule sampling probability
        self.min_value = -1e8
        opt.beta = 1
        self.beta = opt.beta
        if opt.cnn_backend == 'res101':
            self.cnn = resnet(opt, _num_layers=101, _fixed_block=opt.fixed_block, pretrained=True)
        elif opt.cnn_backend == 'res152':
            self.cnn = resnet(opt, _num_layers=152, _fixed_block=opt.fixed_block, pretrained=True)
        elif opt.cnn_backend == 'vgg16':
            self.cnn = vgg16(opt, pretrained=True)

        self.det_fc = nn.Sequential(nn.Embedding(self.detect_size+1, 300),
                                    nn.ReLU(),
                                    nn.Dropout())

        self.loc_fc = nn.Sequential(nn.Linear(5, 300),
                                    nn.ReLU(),
                                    nn.Dropout())

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size+self.detect_size+1, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))

        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))

        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))

        self.pool_embed = nn.Sequential(nn.Linear(self.pool_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))

        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.ctx2pool = nn.Linear(self.rnn_size, self.att_hid_size)

        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.roi_align = RoIAlignAvg(1, 1, 1.0 / self.stride)

        #self.grid_size = 1
        #self.roi_crop = _RoICrop()
        self.critLM = utils.LMCriterion(opt)
        self.critBN = utils.BNCriterion(opt)
        self.critFG = utils.FGCriterion(opt)

        if opt.self_critical:
            print("load reward function...")
            self.get_self_critical_reward = get_self_critical_reward(opt)
            self.critRL = utils.RewardCriterion(opt)

        # initialize the glove weight for the labels.
        self.det_fc[0].weight.data.copy_(opt.glove_clss)
        for p in self.det_fc[0].parameters(): p.requires_grad=False

    def _reinit_word_weight(self, opt, ctoi, wtoi):
        self.det_fc[0].weight.data.copy_(opt.glove_clss)
        for p in self.det_fc[0].parameters(): p.requires_grad=False

        # copy the word embedding
        for w_old, w_new in utils.noc_word_map.items():
            idx_old = ctoi[w_old] + self.vocab_size
            if w_new in ctoi:
                idx_new = ctoi[w_new] + self.vocab_size
            elif w_new in wtoi:
                idx_new = int(wtoi[w_new])
            else:
                pdb.set_trace()
            self.embed[0].weight.data[idx_old].copy_(self.embed[0].weight.data[idx_new])



    def forward(self, img, seq, gt_seq, num, ppls, gt_boxes, mask_boxes, opt, eval_opt = {}):
        if opt == 'MLE':
            return self._forward(img, seq, ppls, gt_boxes, mask_boxes, num)
        elif opt == 'RL':
            # with torch.no_grad():
            greedy_result = self._sample(Variable(img.data, volatile=True), Variable(ppls.data, volatile=True), \
                                    Variable(num.data, volatile=True), {'sample_max':1, 'beam_size':1, 'inference_mode' : False})

            gen_result = self._sample(img, ppls, num, {'sample_max':0, 'beam_size':1, 'inference_mode' : False})
            
            reward, cider_score = self.get_self_critical_reward(gen_result[:3], greedy_result[:3], gt_seq, num[:,0])
            reward = Variable(torch.from_numpy(reward).type_as(img.data).float())
            cider_score = Variable(torch.Tensor(1).fill_(cider_score).type_as(img.data))
            lm_loss, bn_loss, fg_loss = self.critRL(gen_result[0], gen_result[1], gen_result[2], gen_result[3], \
                                                            gen_result[4], gen_result[5], reward)
            return lm_loss, bn_loss, fg_loss, cider_score

        elif opt == 'sample':
            seq, bn_seq, fg_seq, seqLogprobs, bnLogprobs, fgLogprobs = self._sample(img, ppls, num, eval_opt)
            return Variable(seq), Variable(bn_seq), Variable(fg_seq)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))

    def _forward(self, img, seq, ppls, gt_boxes, mask_boxes, num):

        seq = seq.view(-1, seq.size(2), seq.size(3))
        seq_update = seq.data.clone()

        batch_size = img.size(0)
        seq_batch_size = seq.size(0)
        rois_num = ppls.size(1)
        # constructing the mask.

        pnt_mask = mask_boxes.data.new(batch_size, rois_num+1).byte().fill_(1)
        for i in range(batch_size):
            pnt_mask[i,:num.data[i,1]+1] = 0
        pnt_mask = Variable(pnt_mask)

        state = self.init_hidden(seq_batch_size)
        rnn_output = []
        roi_labels = []
        det_output = []

        if self.finetune_cnn:
            conv_feats, fc_feats = self.cnn(img)
        else:
            # with torch.no_grad():
            conv_feats, fc_feats = self.cnn(Variable(img.data, volatile=True))
            conv_feats = Variable(conv_feats.data)
            fc_feats = Variable(fc_feats.data)
        # pooling the conv_feats
        rois = ppls.data.new(batch_size, rois_num, 5)
        rois[:,:,1:] = ppls.data[:,:,:4]

        for i in range(batch_size): rois[i,:,0] = i
        pool_feats = self.roi_align(conv_feats, Variable(rois.view(-1,5)))
        pool_feats = pool_feats.view(batch_size, rois_num, self.att_feat_size)

        loc_input = ppls.data.new(batch_size, rois_num, 5)
        loc_input[:,:,:4] = ppls.data[:,:,:4] / self.image_crop_size
        loc_input[:,:,4] = ppls.data[:,:,5]
        loc_feats = self.loc_fc(Variable(loc_input))

        label_input = seq.data.new(batch_size, rois_num)
        label_input[:,:] = ppls.data[:,:,4]
        label_feat = self.det_fc(Variable(label_input))

        # pool_feats = pool_feats + label_feat
        pool_feats = torch.cat((pool_feats, loc_feats, label_feat), 2)
        # transpose the conv_feats
        conv_feats = conv_feats.view(batch_size, self.att_feat_size, -1).transpose(1,2).contiguous()

        # replicate the feature to map the seq size.
        fc_feats = fc_feats.view(batch_size, 1, self.fc_feat_size)\
                .expand(batch_size, self.seq_per_img, self.fc_feat_size)\
                .contiguous().view(-1, self.fc_feat_size)
        conv_feats = conv_feats.view(batch_size, 1, self.att_size*self.att_size, self.att_feat_size)\
                .expand(batch_size, self.seq_per_img, self.att_size*self.att_size, self.att_feat_size)\
                .contiguous().view(-1, self.att_size*self.att_size, self.att_feat_size)
        pool_feats = pool_feats.view(batch_size, 1, rois_num, self.pool_feat_size)\
                .expand(batch_size, self.seq_per_img, rois_num, self.pool_feat_size)\
                .contiguous().view(-1, rois_num, self.pool_feat_size)
        pnt_mask = pnt_mask.view(batch_size, 1, rois_num+1).expand(batch_size, self.seq_per_img, rois_num+1)\
                    .contiguous().view(-1, rois_num+1)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        conv_feats = self.att_embed(conv_feats)
        pool_feats = self.pool_embed(pool_feats)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_conv_feats = self.ctx2att(conv_feats)
        p_pool_feats = self.ctx2pool(pool_feats)

        # calculate the overlaps between the rois and gt_bbox.
        overlaps = utils.bbox_overlaps(ppls.data, gt_boxes.data)
        for i in range(self.seq_length):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i, 0].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i, 0].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it)
            else:
                it = seq[:, i, 0].clone()
            # break if all the sequences end
            if i >= 1 and seq[:, i, 0].data.sum() == 0:
                break

            roi_label = utils.bbox_target(ppls, mask_boxes[:,:,:,i+1], overlaps, seq[:,i+1], seq_update[:,i+1], self.vocab_size) # roi_label if for the target seq
            # pdb.set_trace()

            roi_labels.append(roi_label.view(seq_batch_size, -1))

            xt = self.embed(it)
            output, det_prob, state = self.core(xt, fc_feats, conv_feats, p_conv_feats, pool_feats, p_pool_feats, pnt_mask, pnt_mask, state)

            rnn_output.append(output)
            det_output.append(det_prob)

        seq_cnt = len(det_output)
        rnn_output = torch.cat([_.unsqueeze(1) for _ in rnn_output], 1)
        det_output = torch.cat([_.unsqueeze(1) for _ in det_output], 1)
        roi_labels = torch.cat([_.unsqueeze(1) for _ in roi_labels], 1)

        det_output = F.log_softmax(det_output, dim=2)
        decoded = F.log_softmax(self.beta * self.logit(rnn_output), dim=2)
        lambda_v = det_output[:,:,0].contiguous()
        prob_det = det_output[:,:,1:].contiguous()

        decoded = decoded+lambda_v.view(seq_batch_size, seq_cnt, 1).expand_as(decoded)
        decoded  = decoded.view((seq_cnt)*seq_batch_size, -1)

        roi_labels = Variable(roi_labels)
        prob_det = prob_det * roi_labels

        roi_cnt = roi_labels.sum(2)
        roi_cnt[roi_cnt==0]=1
        vis_prob = prob_det.sum(2) / roi_cnt
        vis_prob = vis_prob.view(-1,1)
        
        seq_update = Variable(seq_update)
        lm_loss = self.critLM(decoded, vis_prob, seq_update[:,1:seq_cnt+1, 0].clone())
        # do the cascade object recognition.
        vis_idx = seq_update[:,1:seq_cnt+1,0].data - self.vocab_size
        vis_idx[vis_idx<0] = 0
        vis_idx = Variable(vis_idx.view(-1))

        # roi_labels = Variable(roi_labels)
        bn_logprob, fg_logprob = self.ccr_core(vis_idx, pool_feats, rnn_output, roi_labels, seq_batch_size, seq_cnt)

        bn_loss = self.critBN(bn_logprob, seq_update[:,1:seq_cnt+1, 1].clone())
        fg_loss = self.critFG(fg_logprob, seq_update[:,1:seq_cnt+1, 2].clone())

        return lm_loss, bn_loss, fg_loss

    def _sample(self, img, ppls, num, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        inference_mode = opt.get('inference_mode', True)

        batch_size = img.size(0)
        rois_num = ppls.size(1)

        if beam_size > 1 or self.cbs:
            return self._sample_beam(img, ppls, num, opt)

        if self.finetune_cnn:
            conv_feats, fc_feats = self.cnn(img)
        else:
            # with torch.no_grad():
            conv_feats, fc_feats = self.cnn(Variable(img.data, volatile=True))
            conv_feats = Variable(conv_feats.data)
            fc_feats = Variable(fc_feats.data)

        # conv_feats, fc_feats = self.cnn(img)
        rois = ppls.data.new(batch_size, rois_num, 5)
        rois[:,:,1:] = ppls.data[:,:,:4]

        for i in range(batch_size): rois[i,:,0] = i
        pool_feats = self.roi_align(conv_feats, Variable(rois.view(-1,5)))
        pool_feats = pool_feats.view(batch_size, rois_num, self.att_feat_size)

        loc_input = ppls.data.new(batch_size, rois_num, 5)
        loc_input[:,:,:4] = ppls.data[:,:,:4] / self.image_crop_size
        loc_input[:,:,4] = ppls.data[:,:,5]
        loc_feats = self.loc_fc(Variable(loc_input))

        label_input = ppls.data.new(batch_size, rois_num).long()
        label_input[:,:] = ppls.data[:,:,4]
        label_feat = self.det_fc(Variable(label_input))

        # pool_feats = pool_feats + label_feat
        pool_feats = torch.cat((pool_feats, loc_feats, label_feat), 2)
        # transpose the conv_feats
        conv_feats = conv_feats.view(batch_size, self.att_feat_size, -1).transpose(1,2).contiguous()
        # embed fc and att feats
        pool_feats = self.pool_embed(pool_feats)
        fc_feats = self.fc_embed(fc_feats)
        conv_feats = self.att_embed(conv_feats)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_conv_feats = self.ctx2att(conv_feats)
        p_pool_feats = self.ctx2pool(pool_feats)

        vis_offset = (torch.arange(0, batch_size)*rois_num).view(batch_size).type_as(ppls.data).long()
        roi_offset = (torch.arange(0, batch_size)*(rois_num+1)).view(batch_size).type_as(ppls.data).long()

        # constructing the mask.
        pnt_mask = ppls.data.new(batch_size, rois_num+1).byte().fill_(1)
        for i in range(batch_size):
            pnt_mask[i,:num.data[i,1]+1] = 0
        pnt_mask = Variable(pnt_mask)
        pnt_mask_list = []
        pnt_mask_list.append(pnt_mask)

        att_mask = pnt_mask.clone()
        state = self.init_hidden(batch_size)

        seq = []
        seqLogprobs = []
        bn_seq = []
        bnLogprobs = []
        fg_seq = []
        fgLogprobs = []

        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, Variable(it)) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            roi_idx = it.clone() - self.vocab_size - 1 # starting from 0
            roi_mask = roi_idx < 0
            roi_idx_offset = roi_idx + vis_offset
            roi_idx_offset[roi_mask] = 0

            vis_idx = ppls.data[:,:,4].clone().view(-1)[roi_idx_offset].long()
            vis_idx[roi_mask] = 0

            # if inference_mode:
            # if the roi_idx is selected, we need to make sure this is not selected again.
            pnt_idx_offset = roi_idx + roi_offset + 1
            pnt_idx_offset[roi_mask] = 0
            pnt_mask_new = pnt_mask_list[-1].data.clone()
            pnt_mask_new.view(-1)[pnt_idx_offset] = 1
            pnt_mask_new.view(-1)[0] = 0
            pnt_mask_list.append(Variable(pnt_mask_new))

            # tmp_feat = concat_feat.view(-1, self.rnn_size)[tmp_idx_offset]
            # we need to convert the roi index to label index.
            it_new = it.clone()
            it_new[it > self.vocab_size] = (vis_idx[roi_mask==0] + self.vocab_size)
            xt = self.embed(Variable(it_new))

            if t >= 1:
                # do the cascade caption refinement here
                roi_labels = pool_feats.data.new(batch_size*rois_num).zero_()
                if (roi_mask==0).sum() > 0: roi_labels[roi_idx_offset[roi_mask==0]] = 1
                roi_labels = roi_labels.view(batch_size, 1, rois_num)

                bn_logprob, fg_logprob = self.ccr_core(vis_idx, pool_feats, \
                        rnn_output.view(batch_size,1,self.rnn_size), Variable(roi_labels), batch_size, 1)
                bn_logprob = bn_logprob.view(batch_size, -1)
                fg_logprob = fg_logprob.view(batch_size, -1)

                if sample_max:
                    slp_bn, it_bn = torch.max(bn_logprob.data, 1)
                    slp_fg, it_fg = torch.max(fg_logprob.data, 1)
                else:
                    if temperature == 1.0:
                        bn_prob_prev = torch.exp(bn_logprob.data)
                        fg_prob_prev = torch.exp(fg_logprob.data)
                    else:
                        bn_prob_prev = torch.exp(torch.div(bn_logprob.data, temperature))
                        fg_prob_prev = torch.exp(torch.div(fg_logprob.data, temperature))

                    it_bn = torch.multinomial(bn_prob_prev, 1)
                    it_fg = torch.multinomial(fg_prob_prev, 1)

                    slp_bn = bn_logprob.gather(1, Variable(it_bn)) # gather the logprobs at sampled positions
                    slp_fg = fg_logprob.gather(1, Variable(it_fg)) # gather the logprobs at sampled positions

                it_bn[roi_mask] = 0
                it_fg[roi_mask] = 0

                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                # if unfinished.sum() == 0:
                    # break
                    # continue
                it = it * unfinished.type_as(it)
                seq.append(it) #seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))
                bn_seq.append(it_bn)
                bnLogprobs.append(slp_bn.view(-1))
                fg_seq.append(it_fg)
                fgLogprobs.append(slp_fg.view(-1))

            rnn_output, det_prob, state = self.core(xt, fc_feats, conv_feats, p_conv_feats, pool_feats, p_pool_feats, att_mask, pnt_mask_list[-1], state)

            # pnt_mask = pnt_mask_new # update the mask

            det_prob = F.log_softmax(det_prob, dim=1)
            decoded = F.log_softmax(self.beta * self.logit(rnn_output), dim=1)
            lambda_v = det_prob[:,0].contiguous()
            prob_det = det_prob[:,1:].contiguous()

            decoded = decoded + lambda_v.view(batch_size, 1).expand_as(decoded)
            logprobs = torch.cat([decoded, prob_det], 1)
            # logprobs = torch.log(decoded)

        seq = torch.cat([_.unsqueeze(1) for _ in seq], 1)
        seqLogprobs = torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)
        bn_seq = torch.cat([_.unsqueeze(1) for _ in bn_seq], 1)
        bnLogprobs = torch.cat([_.unsqueeze(1) for _ in bnLogprobs], 1)
        fg_seq = torch.cat([_.unsqueeze(1) for _ in fg_seq], 1)
        fgLogprobs = torch.cat([_.unsqueeze(1) for _ in fgLogprobs], 1)


        return seq, bn_seq, fg_seq, seqLogprobs, bnLogprobs, fgLogprobs

    def _sample_beam(self, img, ppls, num, opt={}):
        
        batch_size = ppls.size(0)
        rois_num = ppls.size(1)

        if self.cbs:
            assert batch_size == 1 # cbs only support batch_size == 1 now.
            tag_size = opt.get('tag_size', 3)
            if self.cbs_mode == 'unique': # re-organize the ppls and make non-same at the top.
                unique_idx = []
                unique_clss = []
                for i in range(num.data[0,1]):
                    det_clss = ppls.data[0,i,4]
                    det_confidence = ppls.data[0,i,5]
                    if det_clss not in unique_clss and det_confidence > 0.8:
                        unique_clss.append(det_clss)
                        unique_idx.append(i)
                tag_size = min(len(unique_idx), tag_size)
                for i in range(num.data[0,1]):
                    if i not in unique_idx:
                        unique_idx.append(i)
                if len(unique_idx) > 0:
                    ppls[0] = ppls[0][unique_idx]
            elif self.cbs_mode =='novel': # force decode only novel concept
                novel_idx = []
                novel_clss = []
                for i in range(num.data[0,1]):
                    det_clss = int(ppls.data[0,i,4])
                    det_confidence = ppls.data[0,i,5]
                    if det_clss in utils.noc_index and det_clss not in novel_clss and det_confidence > 0.8:
                        novel_clss.append(det_clss)
                        novel_idx.append(i)
                tag_size = min(len(novel_idx), tag_size)
                for i in range(num.data[0,1]):
                    if i not in novel_idx:
                        novel_idx.append(i)
                if len(novel_idx) > 0:
                    ppls[0] = ppls[0][novel_idx]
            elif self.cbs_mode == 'all':
                tag_size = min(num.data[0,1], tag_size)
            _, tags = utils.cbs_beam_tag(tag_size)
            beam_size = len(tags) * opt.get('beam_size', 3)
            opt['beam_size'] = beam_size
        else:
            beam_size = opt.get('beam_size', 10)

        if self.finetune_cnn:
            conv_feats, fc_feats = self.cnn(img)
        else:
            # with torch.no_grad():
            conv_feats, fc_feats = self.cnn(Variable(img.data, volatile=True))
            conv_feats = Variable(conv_feats.data)
            fc_feats = Variable(fc_feats.data)

        # conv_feats, fc_feats = self.cnn(img)
        rois = ppls.data.new(batch_size, rois_num, 5)
        rois[:,:,1:] = ppls.data[:,:,:4]

        for i in range(batch_size): rois[i,:,0] = i
        pool_feats = self.roi_align(conv_feats, Variable(rois.view(-1,5)))
        pool_feats = pool_feats.view(batch_size, rois_num, self.att_feat_size)

        loc_input = ppls.data.new(batch_size, rois_num, 5)
        loc_input[:,:,:4] = ppls.data[:,:,:4] / self.image_crop_size
        loc_input[:,:,4] = ppls.data[:,:,5]
        loc_feats = self.loc_fc(Variable(loc_input))

        label_input = ppls.data.new(batch_size, rois_num).long()
        label_input[:,:] = ppls.data[:,:,4]
        label_feat = self.det_fc(Variable(label_input))

        # pool_feats = pool_feats + label_feat
        pool_feats = torch.cat((pool_feats, loc_feats, label_feat), 2)
        # transpose the conv_feats
        conv_feats = conv_feats.view(batch_size, self.att_feat_size, -1).transpose(1,2).contiguous()
        # embed fc and att feats
        pool_feats = self.pool_embed(pool_feats)
        fc_feats = self.fc_embed(fc_feats)
        conv_feats = self.att_embed(conv_feats)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_conv_feats = self.ctx2att(conv_feats)
        p_pool_feats = self.ctx2pool(pool_feats)


        # constructing the mask.
        pnt_mask = ppls.data.new(batch_size, rois_num+1).byte().fill_(1)
        for i in range(batch_size):
            pnt_mask[i,:num.data[i,1]+1] = 0
        pnt_mask = Variable(pnt_mask)

        vis_offset = (torch.arange(0, beam_size)*rois_num).view(beam_size).type_as(ppls.data).long()
        roi_offset = (torch.arange(0, beam_size)*(rois_num+1)).view(beam_size).type_as(ppls.data).long()

        seq = ppls.data.new(self.seq_length, batch_size).zero_().long()
        seqLogprobs = ppls.data.new(self.seq_length, batch_size).float()
        bn_seq = ppls.data.new(self.seq_length, batch_size).zero_().long()
        bnLogprobs = ppls.data.new(self.seq_length, batch_size).float()
        fg_seq = ppls.data.new(self.seq_length, batch_size).zero_().long()
        fgLogprobs = ppls.data.new(self.seq_length, batch_size).float()
        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            beam_fc_feats = fc_feats[k:k+1].expand(beam_size, fc_feats.size(1))
            beam_conv_feats = conv_feats[k:k+1].expand(beam_size, conv_feats.size(1), self.rnn_size).contiguous()
            beam_pool_feats = pool_feats[k:k+1].expand(beam_size, rois_num, self.rnn_size).contiguous()
            beam_p_conv_feats = p_conv_feats[k:k+1].expand(beam_size, conv_feats.size(1), self.att_hid_size).contiguous()
            beam_p_pool_feats = p_pool_feats[k:k+1].expand(beam_size, rois_num, self.att_hid_size).contiguous()

            beam_ppls = ppls[k:k+1].expand(beam_size, rois_num, 6).contiguous()
            beam_pnt_mask = pnt_mask[k:k+1].expand(beam_size, rois_num+1).contiguous()

            it = fc_feats.data.new(beam_size).long().zero_()
            xt = self.embed(Variable(it))

            rnn_output, det_prob, state = self.core(xt, beam_fc_feats, beam_conv_feats, beam_p_conv_feats, \
                                                            beam_pool_feats, beam_p_pool_feats, beam_pnt_mask, beam_pnt_mask, state)

            if self.cbs:
                self.done_beams[k] = self.constraint_beam_search(state, rnn_output, det_prob, beam_fc_feats, beam_conv_feats, beam_p_conv_feats, \
                                                    beam_pool_feats, beam_p_pool_feats, beam_ppls, beam_pnt_mask, vis_offset, roi_offset, tag_size, tags, opt)

                ii = 0
                seq[:, k] = self.done_beams[k][tags[-ii-1]][0]['seq'].cuda() # the first beam has highest cumulative score
                seqLogprobs[:, k] = self.done_beams[k][tags[-ii-1]][0]['logps'].cuda()

                bn_seq[:,k] = self.done_beams[k][tags[-ii-1]][0]['bn_seq'].cuda()
                bnLogprobs[:,k] = self.done_beams[k][tags[-ii-1]][0]['bn_logps'].cuda()

                fg_seq[:,k] = self.done_beams[k][tags[-ii-1]][0]['fg_seq'].cuda()
                fgLogprobs[:,k] = self.done_beams[k][tags[-ii-1]][0]['fg_logps'].cuda()
                    # break

            else:
                self.done_beams[k] = self.beam_search(state, rnn_output, det_prob, beam_fc_feats, beam_conv_feats, beam_p_conv_feats, \
                                                    beam_pool_feats, beam_p_pool_feats, beam_ppls, beam_pnt_mask, vis_offset, roi_offset, opt)
                
                seq[:, k] = self.done_beams[k][0]['seq'].cuda() # the first beam has highest cumulative score
                seqLogprobs[:, k] = self.done_beams[k][0]['logps'].cuda()

                bn_seq[:,k] = self.done_beams[k][0]['bn_seq'].cuda()
                bnLogprobs[:,k] = self.done_beams[k][0]['bn_logps'].cuda()

                fg_seq[:,k] = self.done_beams[k][0]['fg_seq'].cuda()
                fgLogprobs[:,k] = self.done_beams[k][0]['fg_logps'].cuda()

                
        return seq.t(), bn_seq.t(), fg_seq.t(), seqLogprobs.t(), bnLogprobs.t(), fgLogprobs.t()
