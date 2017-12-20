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
from six.moves import cPickle

import opts
import misc.eval_utils
import misc.utils as utils
import misc.model as model
from misc.dataloader import DataLoader
import yaml

# from misc.rewards import get_self_critical_reward
import torchvision.transforms as transforms
import pdb

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None

def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

opt = opts.parse_opt()

if not os.path.exists(opt.checkpoint_path):
    os.makedirs(opt.checkpoint_path)

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
    cider_temp = 0
    start = time.time()

    for step in range(len(dataloader)):
        data = data_iter.next()
        img, iseq, gts_seq, ncap, img_id = data

        iseq = iseq.view(-1, iseq.size(2))

        input_imgs.data.resize_(img.size()).copy_(img)
        input_seqs.data.resize_(iseq.size()).copy_(iseq)
        gt_seqs.data.resize_(gts_seq.size()).copy_(gts_seq)
        num_cap.data.resize_(ncap.size()).copy_(ncap)

        if opt.self_critical:
            loss, cider_score = model(input_imgs, input_seqs, gt_seqs, num_cap, 'RL')         
            cider_temp += cider_score.data[0]
        else:
            loss = model(input_imgs, input_seqs, gt_seqs, num_cap, 'MLE')

        model.zero_grad()
        loss.backward()
        loss_temp += loss.data[0]

        utils.clip_gradient(optimizer, opt.grad_clip)        
        optimizer.step()

        if step % opt.disp_interval == 0 and step != 0:
            end = time.time()
            loss_temp /= opt.disp_interval
            cider_temp /= opt.disp_interval
            print("step {}/{} (epoch {}), loss = {:.3f}, cider_score = {:.3f}, lr = {:.5f}, time/batch = {:.3f}" \
                .format(step, len(dataloader), epoch, loss_temp, cider_temp, opt.learning_rate, end - start))
            start = time.time()

            loss_temp = 0
            cider_temp = 0

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            if tf is not None:
                add_summary_value(tf_summary_writer, 'train_loss', train_loss, iteration)
                add_summary_value(tf_summary_writer, 'learning_rate', opt.learning_rate, iteration)
                add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                if opt.self_critical:
                    add_summary_value(tf_summary_writer, 'cider_score', cider_score.data[0], iteration)
                tf_summary_writer.flush()

            loss_history[iteration] = loss.data[0]
            lr_history[iteration] = opt.learning_rate
            ss_prob_history[iteration] = model.ss_prob


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
        seq, _ = model(input_imgs, input_seqs, gt_seqs, num_cap, 'sample')

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

    # Write validation result into summary
    if tf is not None:
        for k,v in lang_stats.items():
            add_summary_value(tf_summary_writer, k, v, iteration)
        tf_summary_writer.flush()
    val_result_history[iteration] = {'lang_stats': lang_stats, 'predictions': predictions}

    return lang_stats

####################################################################################
# Main
####################################################################################
# initialize the data holder.

if __name__ == '__main__':

    input_imgs = torch.FloatTensor(1)
    input_seqs = torch.LongTensor(1)
    gt_seqs = torch.LongTensor(1)
    num_cap = torch.LongTensor(1)

    if opt.cuda:
        input_imgs = input_imgs.cuda()
        input_seqs = input_seqs.cuda()
        gt_seqs = gt_seqs.cuda()
        num_cap = num_cap.cuda()

    input_imgs = Variable(input_imgs)
    input_seqs = Variable(input_seqs)
    gt_seqs = Variable(gt_seqs)
    num_cap = Variable(num_cap)

    tf_summary_writer = tf and tf.summary.FileWriter(opt.checkpoint_path)
    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)

    iteration = infos.get('iter', 0)
    start_epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

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
    for epoch in range(start_epoch, opt.max_epochs):
        
        if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
            if (epoch - opt.learning_rate_decay_start) % opt.learning_rate_decay_every == 0:
                # decay the learning rate.
                opt.learning_rate  = utils.decay_lr(optimizer, opt.learning_rate_decay_rate)

        if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
            model.ss_prob = opt.ss_prob

        train(epoch, opt)
        lang_stats = eval(opt)

        # Save model if is improving on validation result
        if opt.language_eval == 1:
            current_score = lang_stats['CIDEr']
        else:
            current_score = - val_loss
        
        best_flag = False

        if best_val_score is None or current_score > best_val_score:
            best_val_score = current_score
            best_flag = True
        checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print("model saved to {}".format(checkpoint_path))
        optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
        torch.save(optimizer.state_dict(), optimizer_path)

        # Dump miscalleous informations
        infos['iter'] = iteration
        infos['epoch'] = epoch
        infos['best_val_score'] = best_val_score
        infos['opt'] = opt
        infos['vocab'] = dataset.itow

        histories['val_result_history'] = val_result_history
        histories['loss_history'] = loss_history
        histories['lr_history'] = lr_history
        histories['ss_prob_history'] = ss_prob_history
        with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
            cPickle.dump(infos, f)
        with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
            cPickle.dump(histories, f)

        if best_flag:
            checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print("model saved to {}".format(checkpoint_path))
            with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                cPickle.dump(infos, f)

