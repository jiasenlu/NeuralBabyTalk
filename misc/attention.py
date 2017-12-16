
import math
import torch
import warnings

import torch
import torch.nn as nn 
import torch.nn.functional as F
import pdb

class AdaAtt_attention(nn.Module):
    def __init__(self, rnn_size, att_hid_size, grid_num, dropout):
        super(AdaAtt_attention, self).__init__()
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.att_hid_size = att_hid_size
        self.grid_num = grid_num

        # fake region embed
        self.f_fc1 = nn.Linear(self.rnn_size, self.rnn_size)
        self.f_fc2 = nn.Linear(self.rnn_size, self.att_hid_size)

        # h out embed
        self.h_fc1 = nn.Linear(self.rnn_size, self.rnn_size)
        self.h_fc2 = nn.Linear(self.rnn_size, self.att_hid_size)

        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.att2h = nn.Linear(self.rnn_size, self.rnn_size)

        self.inplace = False

    def forward(self, h_out, fake_region, conv_feat, conv_feat_embed):

        # View into three dimensions
        conv_feat = conv_feat.view(-1, self.grid_num, self.rnn_size)
        conv_feat_embed = conv_feat_embed.view(-1, self.grid_num, self.att_hid_size)

        # view neighbor from bach_size * neighbor_num x rnn_size to bach_size x rnn_size * neighbor_num
        fake_region = F.dropout(F.relu(self.f_fc1(fake_region.view(-1, self.rnn_size)), inplace=self.inplace), 
                                self.dropout, training=self.training, inplace=self.inplace)
        
        fake_region_embed = self.f_fc2(fake_region)

        h_out_linear = F.dropout(F.tanh(self.h_fc1(h_out.view(-1, self.rnn_size))),
                                self.dropout, training=self.training, inplace=self.inplace)
        
        h_out_embed = self.h_fc2(h_out_linear)

        img_all = torch.cat([fake_region.view(-1,1,self.rnn_size), conv_feat], 1)
        img_all_embed = torch.cat([fake_region_embed.view(-1,1,self.rnn_size), conv_feat_embed], 1)

        hA = F.tanh(img_all_embed + h_out_embed.view(-1,1,self.att_hid_size))
        hA = F.dropout(hA, self.dropout, self.training, inplace=self.inplace)
        
        hAflat = self.alpha_net(hA.view(-1, self.att_hid_size))
        PI = F.softmax(hAflat.view(-1, self.grid_num + 1))

        visAtt = torch.bmm(PI.view(-1, 1, self.grid_num+1), img_all)
        visAtt = visAtt.view(-1, self.rnn_size)
        attn_feat = visAtt + h_out_linear
        attn_feat = F.tanh(self.att2h(attn_feat))
        attn_feat = F.dropout(attn_feat, self.dropout, self.training, inplace=self.inplace)
        
        return attn_feat