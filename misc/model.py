from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
from misc.resnet import resnet
from misc.CaptionModel import CaptionModel
from misc.rewards import get_self_critical_reward

# from misc.adaptiveLSTMCell import adaptiveLSTMCell
import pdb

class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size    
        self.seq_per_img = 5
        self.att_size = opt.att_size
        self.beta = 3
        
        self.ss_prob = 0.0 # Schedule sampling probability
        self.cnn = resnet(_num_layers=101, _fixed_block=opt.fixed_block, pretrained=True)

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))

        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))

        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))

        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)


        self.critLM = utils.LMCriterion()

        if opt.self_critical:
            print("load reward function...")
            self.get_self_critical_reward = get_self_critical_reward()
            self.critRL = utils.RewardCriterion()

    def forward(self, img, seq, gt_seq, ncap, opt):
        if opt == 'MLE':
            return self._forward(img, seq)
        elif opt == 'RL':
            gen_result = self._sample(img, {'sample_max':0})
            greedy_result = self._sample(Variable(img.data, volatile=True), {'sample_max':0})

            reward, cider_score = self.get_self_critical_reward(gen_result[0], greedy_result[0], gt_seq, ncap)
            reward = Variable(torch.from_numpy(reward).type_as(img.data).float())
            cider_score = Variable(torch.Tensor(1).fill_(cider_score).type_as(img.data))

            loss = self.critRL(gen_result[1], gen_result[0], reward)

            return loss, cider_score

        elif opt == 'sample':
            eval_opt = {'sample_max':1, 'beam_size': 1}
            return self._sample(img, eval_opt)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))

    def _forward(self, img, seq):

        batch_size = img.size(0)
        seq_batch_size = seq.size(0)
        
        state = self.init_hidden(seq_batch_size)
        
        outputs = []

        conv_feats, fc_feats = self.cnn(img)
        # transpose the conv_feats
        conv_feats = conv_feats.view(batch_size, self.att_feat_size, -1).transpose(1,2).contiguous()
        # replicate the feature to map the seq size.
        fc_feats = fc_feats.view(batch_size, 1, self.fc_feat_size)\
                .expand(batch_size, self.seq_per_img, self.fc_feat_size)\
                .contiguous().view(-1, self.fc_feat_size)
        conv_feats = conv_feats.view(batch_size, 1, self.att_size*self.att_size, self.att_feat_size)\
                .expand(batch_size, self.seq_per_img, self.att_size*self.att_size, self.att_feat_size)\
                .contiguous().view(-1, self.att_size*self.att_size, self.att_feat_size)   

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        conv_feats = self.att_embed(conv_feats)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_conv_feats = self.ctx2att(conv_feats)

        for i in range(self.seq_length):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            xt = self.embed(it)

            output, state = self.core(xt, fc_feats, conv_feats, p_conv_feats, state)
            output = F.log_softmax(self.logit(output), dim=1)
            outputs.append(output)

        outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1)

        loss = self.critLM(outputs, seq[:,1:outputs.size(1)+1])

        return loss

    def _sample(self, img, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)

        batch_size = img.size(0)
        state = self.init_hidden(batch_size)


        conv_feats, fc_feats = self.cnn(img)
        # transpose the conv_feats
        conv_feats = conv_feats.view(batch_size, self.att_feat_size, -1).transpose(1,2).contiguous()

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        conv_feats = self.att_embed(conv_feats)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_conv_feats = self.ctx2att(conv_feats)

        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False)) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            xt = self.embed(Variable(it, requires_grad=False))

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it) #seq[t] the input of t+2 time step

                seqLogprobs.append(sampleLogprobs.view(-1))

            output, state = self.core(xt, fc_feats, conv_feats, p_conv_feats, state)
            logprobs = F.log_softmax(self.logit(output), dim=1)

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res


class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats)

        lang_lstm_input = torch.cat([att, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state

class TopDownModel(AttModel):
    def __init__(self, opt):
        super(TopDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownCore(opt)