from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
from PIL import Image
from torchvision.datasets.folder import default_loader
import torch
import torch.utils.data as data

import pdb


class DataLoader(data.Dataset):
    def __init__(self, opt, split='train', seq_per_img=5, transform=None):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img
        self.seq_length = opt.seq_length
        self.split = split
        self.seq_per_img = seq_per_img
        self.transform = transform

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_dic)
        self.info = json.load(open(self.opt.input_dic))
        self.itow = self.info['ix_to_word']
        self.wtoi = {w:i for i,w in self.itow.items()}
        self.vocab_size = len(self.itow) + 1 # since it start from 1
        print('vocab size is ', self.vocab_size)

        # open the caption json file
        print('DataLoader loading json file: ', opt.input_json)
        self.caption_file = json.load(open(self.opt.input_json))

        # separate out indexes for each of the provided splits
        self.split_ix = []
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == split:
                self.split_ix.append(ix)

        print('assigned %d images to split %s' %(len(self.split_ix), split))

    def __getitem__(self, index):

        ix = self.split_ix[index]
        # fetch the captions
        captions = self.caption_file[ix]

        ncap = len(captions) # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        # convert caption into sequence label.
        cap_seq = np.zeros([ncap, self.seq_length])
        for i, caption in enumerate(captions):
            for j, w in enumerate(caption):
                if j < self.seq_length:
                    cap_seq[i,j] = self.wtoi[w]

        seq = np.zeros([self.seq_per_img, self.seq_length])
        if ncap < self.seq_per_img:
            # we need to subsample (with replacement)
            for q in range(self.seq_per_img):
                ixl = random.randint(0,ncap)
                seq[q,:] = cap_seq[ixl]
        else:
            ixl = random.randint(0, ncap - self.seq_per_img)
            seq = cap_seq[ixl:ixl+self.seq_per_img]

        input_seq = np.zeros([self.seq_per_img, self.seq_length+1])
        input_seq[:,1:] = seq

        gt_seq = np.zeros([10, self.seq_length])
        gt_seq[:ncap,:] = cap_seq

        # load image here.
        image_id = self.info['images'][ix]['id']
        file_path = self.info['images'][ix]['file_path']

        img = default_loader(os.path.join(self.opt.image_path, file_path))
        img = self.transform(img)

        input_seq = torch.from_numpy(input_seq).long()
        gt_seq = torch.from_numpy(gt_seq).long()

        return img, input_seq, gt_seq, ncap, image_id

    def __len__(self):
        return len(self.split_ix)
