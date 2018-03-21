from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
from misc.model import AttModel
from torch.nn.parameter import Parameter
import pdb

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net =  nn.Linear(self.att_hid_size, 1)
        self.min_value = -1e8
        # self.batch_norm = nn.BatchNorm1d(self.rnn_size)

    def forward(self, h, att_feats, p_att_feats):
        # The p_att_feats here is already projected
        batch_size = h.size(0)
        att_size = att_feats.numel() // batch_size // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                # batch * att_size * att_hid_size
        dot = F.tanh(dot)                              # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        # dot = F.dropout(dot, 0.3, training=self.training)
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size
        # att_res = self.batch_norm(att_res)

        return att_res


class Attention2(nn.Module):
    def __init__(self, opt):
        super(Attention2, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.min_value = -1e8
        # self.batch_norm = nn.BatchNorm1d(self.rnn_size)

    def forward(self, h, att_feats, p_att_feats, mask):
        # The p_att_feats here is already projected
        batch_size = h.size(0)
        att_size = att_feats.numel() // batch_size // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                # batch * att_size * att_hid_size
        dot = F.tanh(dot)                              # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        # dot = F.dropout(dot, 0.3, training=self.training)
        hAflat = self.alpha_net(dot)                           # (batch * att_size) * 1
        hAflat = hAflat.view(-1, att_size)                        # batch * att_size
        hAflat.masked_fill_(mask, self.min_value)
        
        weight = F.softmax(hAflat, dim=1)                             # batch * att_size
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size
        # att_res = self.batch_norm(att_res)

        return att_res

class adaPnt(nn.Module):
    def __init__(self, conv_size, rnn_size, att_hid_size, dropout, min_value, beta):
        super(adaPnt, self).__init__()
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.att_hid_size = att_hid_size
        self.min_value = min_value
        self.conv_size = conv_size

        # fake region embed
        self.f_fc1 = nn.Linear(self.rnn_size, self.rnn_size)
        self.f_fc2 = nn.Linear(self.rnn_size, self.att_hid_size)
        # h out embed
        self.h_fc1 = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.inplace = False
        self.beta = beta
    def forward(self, h_out, fake_region, conv_feat, conv_feat_embed, mask):

        batch_size = h_out.size(0)
        # View into three dimensions
        # conv_feat = conv_feat.view(batch_size, -1, self.conv_size)
        roi_num = conv_feat_embed.size(1)
        conv_feat_embed = conv_feat_embed.view(-1, roi_num, self.att_hid_size)
        # view neighbor from bach_size * neighbor_num x rnn_size to bach_size x rnn_size * neighbor_num
        fake_region = F.relu(self.f_fc1(fake_region.view(-1, self.rnn_size)), inplace=self.inplace)
        fake_region_embed = self.f_fc2(fake_region)
        # fake_region_embed = self.f_fc1(fake_region.view(-1, self.rnn_size))
        h_out_embed = self.h_fc1(h_out)
        # img_all = torch.cat([fake_region.view(-1,1,self.conv_size), conv_feat], 1)
        img_all_embed = torch.cat([fake_region_embed.view(-1,1,self.att_hid_size), conv_feat_embed], 1)
        hA = F.tanh(img_all_embed + h_out_embed.view(-1,1,self.att_hid_size))
        # hA = F.dropout(hA, 0.3, self.training)
        hAflat = self.alpha_net(hA.view(-1, self.att_hid_size))
        hAflat = hAflat.view(-1, roi_num + 1)
        hAflat.masked_fill_(mask, self.min_value)
        # det_prob = F.log_softmax(hAflat, dim=1)
        return hAflat

class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.min_value = -1e8

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size) # we, fc, h^2_t-1

        # self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size*2, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)
        self.attention2 = Attention2(opt)

        self.adaPnt = adaPnt(opt.input_encoding_size, opt.rnn_size, opt.att_hid_size, self.drop_prob_lm, self.min_value, opt.beta)
        self.i2h_2 = nn.Linear(opt.rnn_size*2, opt.rnn_size)
        self.h2h_2 = nn.Linear(opt.rnn_size, opt.rnn_size)

    def forward(self, xt, fc_feats, conv_feats, p_conv_feats, pool_feats, p_pool_feats, att_mask, pnt_mask, state):
        
        prev_h = state[0][-1]
        # att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)
        att_lstm_input = torch.cat([fc_feats, xt], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))
        att = self.attention(h_att, conv_feats, p_conv_feats)
        att2 = self.attention2(h_att, pool_feats, p_pool_feats, att_mask[:,1:])
        lang_lstm_input = torch.cat([att+att2, h_att], 1)

        ada_gate_point = F.sigmoid(self.i2h_2(lang_lstm_input) + self.h2h_2(state[0][1]))
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))
        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        fake_box = F.dropout(ada_gate_point*F.tanh(state[1][1]), self.drop_prob_lm, training=self.training)
        det_prob = self.adaPnt(output, fake_box, pool_feats, p_pool_feats, pnt_mask)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))
        return output, det_prob, state

class Att2in2Core(nn.Module):
    def __init__(self, opt):
        super(Att2in2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        #self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_hid_size = opt.att_hid_size
        self.min_value = -1e8

        self.adaPnt = adaPnt(opt.input_encoding_size, opt.rnn_size, opt.att_hid_size, self.drop_prob_lm, self.min_value, opt.beta)

        # Build a LSTM
        self.a2c1 = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.a2c2 = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 6 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 6 * self.rnn_size)
        self.dropout1 = nn.Dropout(self.drop_prob_lm)
        self.dropout2 = nn.Dropout(self.drop_prob_lm)
        self.attention = Attention(opt)
        self.attention2 = Attention2(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, pool_feats, p_pool_feats, att_mask, pnt_mask, state):

        att_res1 = self.attention(state[0][-1], att_feats, p_att_feats)
        att_res2 = self.attention2(state[0][-1], pool_feats, p_pool_feats, att_mask[:,1:])

        # xt_input = torch.cat([fc_feats, xt], 1)
        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 4 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
        s_gate = sigmoid_chunk.narrow(1, self.rnn_size * 3, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 4 * self.rnn_size, 2 * self.rnn_size) + \
            self.a2c1(att_res1) + self.a2c2(att_res2)

        in_transform = torch.max(\
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)
        fake_box = s_gate * F.tanh(next_c)

        output = self.dropout1(next_h)
        fake_box = self.dropout2(fake_box)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        det_prob = self.adaPnt(output, fake_box, pool_feats, p_pool_feats, pnt_mask)
        return output, det_prob, state

class TopDownModel(AttModel):
    def __init__(self, opt):
        super(TopDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownCore(opt)
        self.ccr_core = CascadeCore(opt)

class Att2in2Model(AttModel):
    def __init__(self, opt):
        super(Att2in2Model, self).__init__(opt)
        self.num_layers = 1
        self.core = Att2in2Core(opt)
        self.ccr_core = CascadeCore(opt)

class CascadeCore(nn.Module):
    def __init__(self, opt):
        super(CascadeCore, self).__init__()
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.fg_size = opt.fg_size
        self.fg_size = opt.fg_size

        self.bn_fc = nn.Sequential(nn.Linear(opt.rnn_size+opt.rnn_size, opt.rnn_size),
                                nn.ReLU(),
                                nn.Dropout(opt.drop_prob_lm),
                                nn.Linear(opt.rnn_size, 2))

        self.fg_fc = nn.Sequential(nn.Linear(opt.rnn_size+opt.rnn_size, opt.rnn_size),
                                nn.ReLU(),
                                nn.Dropout(opt.drop_prob_lm),
                                nn.Linear(opt.rnn_size, 300))

        # initialize the fine-grained glove embedding.
        self.fg_emb = Parameter(opt.glove_fg)
        self.fg_emb.requires_grad=False

        # setting the fg mask for the cascadeCore.
        self.fg_mask = Parameter(opt.fg_mask)
        self.fg_mask.requires_grad=False
        self.min_value = -1e8
        self.beta = opt.beta

    def forward(self, fg_idx, pool_feats, rnn_outs, roi_labels, seq_batch_size, seq_cnt):
        
        roi_num = pool_feats.size(1)
        pool_feats = pool_feats.view(seq_batch_size, 1, roi_num, self.rnn_size) * \
                    roi_labels.view(seq_batch_size, seq_cnt, roi_num, 1)
        
        # get the average of the feature. # size:  seq_batch_size, seq_cnt, rnn_size.
        pool_cnt = roi_labels.sum(2)
        pool_cnt[pool_cnt==0] = 1
        pool_feats = pool_feats.sum(2) / pool_cnt.view(seq_batch_size, seq_cnt, 1)

        # concate with the rnn_output feature.
        pool_feats = torch.cat((rnn_outs, pool_feats), 2)
        bn_logprob = F.log_softmax(self.bn_fc(pool_feats), dim=2)
        
        fg_out = self.fg_fc(pool_feats)
        # construct the mask for finegrain classification.
        # fg_out 
        fg_score = torch.mm(fg_out.view(-1,300), self.fg_emb.t()).view(seq_batch_size, -1, self.fg_size+1)

        fg_mask = self.fg_mask[fg_idx]
        fg_score.masked_fill_(fg_mask.view_as(fg_score), self.min_value)
        fg_logprob = F.log_softmax(self.beta * fg_score, dim=2)

        return bn_logprob, fg_logprob
