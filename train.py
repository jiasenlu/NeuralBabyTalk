from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import time
import os

import opts
import misc.eval_utils
import misc.utils as utils
import misc.model as model
from misc.dataloader import DataLoader
import yaml

# from misc.rewards import get_self_critical_reward
import torchvision.transforms as transforms
import pdb

opt = opts.parse_opt()
# with open(opt.path_opt, 'r') as handle:
#     options_yaml = yaml.load(handle)

####################################################################################
# Data Loader
####################################################################################
dataset = DataLoader(opt, split='train',
                    transform=transforms.Compose([
                        transforms.Resize(opt.image_size),
                        transforms.RandomCrop(opt.image_crop_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], 
                                            [0.229, 0.224, 0.225])
                        ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                        shuffle=True, num_workers=opt.num_workers)

dataset_val = DataLoader(opt, split='val',
                    transform=transforms.Compose([
                        transforms.Resize(opt.image_size),
                        transforms.RandomCrop(opt.image_crop_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], 
                                            [0.229, 0.224, 0.225])
                        ]))
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size,
                                        shuffle=False, num_workers=opt.num_workers)

####################################################################################
# Build the Model
####################################################################################
opt.vocab_size = dataset.vocab_size
opt.seq_length = opt.seq_length
if not opt.finetune_cnn: opt.fixed_block = 4 # if not finetune, fix all cnn block

model = model.TopDownModel(opt)
critLM = utils.LMCriterion()
# critRL = utils.RewardCriterion()

if opt.mGPUs:
    model = nn.DataParallel(model)

if opt.cuda:
    model.cuda()

def train(epoch, opt):
    model.train()

    #########################################################################################
    # Training begins here
    #########################################################################################
    data_iter = iter(dataloader)
    loss_temp = 0
    start = time.time()

    for step in range(len(dataloader)):
        data = data_iter.next()
        img, iseq, gts_seq, ncap, img_id = data

        iseq = iseq.view(-1, iseq.size(2))
        input_imgs.data.resize_(img.size()).copy_(img)
        input_seqs.data.resize_(iseq.size()).copy_(iseq)

        if opt.self_critical:
            loss = model(input_imgs, input_seqs, 'RL')            
        else:
            loss = model(input_imgs, input_seqs, 'MLE')
            reward_ave = 0

        model.zero_grad()
        loss.backward()
        loss_temp += loss.data[0]

        utils.clip_gradient(optimizer, opt.grad_clip)        
        optimizer.step()

        if step % opt.disp_interval == 0 and step != 0:
            end = time.time()
            loss_temp /= opt.disp_interval
            print("step {}/{} (epoch {}), loss = {:.3f}, avg_reward = {:.3f}, time/batch = {:.3f}" \
                .format(step, len(dataloader), epoch, loss_temp, reward_ave, end - start))
            start = time.time()  

def eval(opt):
    model.eval()
    hidden = model.init_hidden(opt.batch_size)
    hidden1 = model.init_hidden(opt.batch_size)

    sample_hidden = model.init_hidden(opt.batch_size)
    sample_hidden1 = model.init_hidden(opt.batch_size)

    #########################################################################################
    # eval begins here
    #########################################################################################
    data_iter_val = iter(dataloader_val)
    loss_temp = 0
    start = time.time()

    num_show = 0
    predictions = []

    for step in range(int(opt.val_images_use / opt.batch_size)):
        data = data_iter_val.next()
        img, iseq, gts_seq, ncap, img_id = data
        iseq = iseq.view(-1, iseq.size(2))

        input_imgs.data.resize_(img.size()).copy_(img)

        eval_opt = {'sample_max':1, 'beam_size':1}
        seq, _ = model(input_imgs, iseq, 'sample')

        sents = utils.decode_sequence(dataset.itow, seq, opt.vocab_size)

        for k, sent in enumerate(sents):
            entry = {'image_id': img_id[k], 'caption': sent}
            predictions.append(entry)
            if num_show < 20:
                print('image %s: %s' %(entry['image_id'], entry['caption']))
                num_show += 1

    lang_stats = None
    if opt.language_eval == 1:
        lang_stats = utils.language_eval(dataset, predictions, str(1), 'val')

####################################################################################
# Main
####################################################################################
# initialize the data holder.
input_imgs = torch.FloatTensor(1)
input_seqs = torch.LongTensor(1)

if opt.cuda:
    input_imgs = input_imgs.cuda()
    input_seqs = input_seqs.cuda()

input_imgs = Variable(input_imgs)
input_seqs = Variable(input_seqs)

params = []
for key, value in dict(model.named_parameters()).items():
    if value.requires_grad:
        if 'cnn' in key:
            params += [{'params':[value], 'lr':opt.cnn_learning_rate, 
                    'weight_decay':opt.cnn_weight_decay, 'betas':(opt.cnn_optim_alpha, opt.cnn_optim_beta)}]
        else:
            params += [{'params':[value], 'lr':opt.learning_rate, 
                'weight_decay':opt.weight_decay, 'betas':(opt.optim_alpha, opt.optim_beta)}]

optimizer = optim.Adam(params)

start_epoch = 0
# start training.

for epoch in range(start_epoch, opt.max_epochs):
    
    # # Assign the learning rate
    # if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
    #     frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
    #     decay_factor = opt.learning_rate_decay_rate  ** frac
    #     utils.set_lr(optimizer, decay_factor) # set the decayed rate
    #     opt.current_lr = opt.
    # else:
    #     opt.current_lr = opt.learning_rate
    # # Assign the scheduled sampling prob
    # if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
    #     frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
    #     opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
    #     model.ss_prob = opt.ss_prob

    # train(epoch, opt)
    eval(opt)
