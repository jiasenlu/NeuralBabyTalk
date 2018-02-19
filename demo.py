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
import torch.backends.cudnn as cudnn
import yaml

import opts
import misc.eval_utils
import misc.utils as utils
import misc.AttModel as AttModel
import yaml

# from misc.rewards import get_self_critical_reward
import torchvision.transforms as transforms
import pdb
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


def demo(opt):
    model.eval()
    #########################################################################################
    # eval begins here
    #########################################################################################
    data_iter_val = iter(dataloader_val)
    loss_temp = 0
    start = time.time()

    num_show = 0
    predictions = []
    count = 0
    for step in range(len(dataloader_val)):
        data = data_iter_val.next()
        img, iseq, gts_seq, num, proposals, bboxs, box_mask, img_id = data

        proposals = proposals[:,:max(int(max(num[:,1])),1),:]
        input_imgs.data.resize_(img.size()).copy_(img)
        input_seqs.data.resize_(iseq.size()).copy_(iseq)
        gt_seqs.data.resize_(gts_seq.size()).copy_(gts_seq)
        input_num.data.resize_(num.size()).copy_(num)
        input_ppls.data.resize_(proposals.size()).copy_(proposals)
        gt_bboxs.data.resize_(bboxs.size()).copy_(bboxs)
        mask_bboxs.data.resize_(box_mask.size()).copy_(box_mask)
        input_imgs.data.resize_(img.size()).copy_(img)

        eval_opt = {'sample_max':1, 'beam_size':3}
        seq, bn_seq, fg_seq, _, _, _ = model._sample(input_imgs, input_ppls, input_num, eval_opt)

        sents, det_idx, det_word = utils.decode_sequence_det(dataset_val.itow, dataset_val.itod, dataset_val.ltow, dataset_val.itoc, dataset_val.wtod, \
                                                            seq, bn_seq, fg_seq, opt.vocab_size, opt)

        im2show = Image.open(os.path.join(opt.image_path, 'val2014/COCO_val2014_%012d.jpg' % img_id[0])).convert('RGB')
        w, h = im2show.size

        # for visulization
        cls_dets = proposals[0][det_idx].numpy()
        cls_dets[:,0] = cls_dets[:,0] * w / float(opt.image_crop_size)
        cls_dets[:,2] = cls_dets[:,2] * w / float(opt.image_crop_size)
        cls_dets[:,1] = cls_dets[:,1] * h / float(opt.image_crop_size)
        cls_dets[:,3] = cls_dets[:,3] * h / float(opt.image_crop_size)

        # fig = plt.figure()
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left=0,right=0,bottom=0,top=0)

        plt.imshow(im2show)
        for i in range(len(cls_dets)):
            ax = utils.vis_detections(ax, dataset_val.itoc[int(cls_dets[i,4])], cls_dets[i,:5], thresh=0)
        plt.axis('off')
        plt.axis('tight')
        plt.tight_layout()
        fig.savefig('/visu/%d.jpg' %(img_id[0]), bbox_inches='tight', pad_inches=0, dpi=150)

####################################################################################
# Main
####################################################################################
# initialize the data holder.
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_from', type=str, default='', help='')
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')     
    parser.add_argument('--id', type=str, default='',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--image_path', type=str, default='/home/jiasen/data/coco/images/',
                    help='path to the h5file containing the image data') 
    args = parser.parse_args()

    infos = {}
    histories = {}
    if args.start_from is not None:
        if args.load_best_score == 1:
            model_path = os.path.join(args.start_from, 'model-best.pth')
            info_path = os.path.join(args.start_from, 'infos_'+args.id+'-best.pkl')
        else:
            model_path = os.path.join(args.start_from, 'model.pth')
            info_path = os.path.join(args.start_from, 'infos_'+args.id+'.pkl')

            # open old infos and check if models are compatible
        with open(info_path) as f:
            infos = cPickle.load(f)
            opt = infos['opt']
            opt.image_path = args.image_path
    else:
        print("please specify the model path...")
        pdb.set_trace()

    cudnn.benchmark = True

    if opt.dataset == 'flickr30k':
        from misc.dataloader_flickr30k import DataLoader
    else:
        from misc.dataloader_coco import DataLoader

    ####################################################################################
    # Data Loader
    ####################################################################################
    dataset_val = DataLoader(opt, split=opt.val_split)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1,
                                            shuffle=False, num_workers=opt.num_workers)

    input_imgs = torch.FloatTensor(1)
    input_seqs = torch.LongTensor(1)
    input_ppls = torch.FloatTensor(1)
    gt_bboxs = torch.FloatTensor(1)
    mask_bboxs = torch.ByteTensor(1)
    gt_seqs = torch.LongTensor(1)
    input_num = torch.LongTensor(1)

    if opt.cuda:
        input_imgs = input_imgs.cuda()
        input_seqs = input_seqs.cuda()
        gt_seqs = gt_seqs.cuda()
        input_num = input_num.cuda()
        input_ppls = input_ppls.cuda()
        gt_bboxs = gt_bboxs.cuda()
        mask_bboxs = mask_bboxs.cuda()

    input_imgs = Variable(input_imgs)
    input_seqs = Variable(input_seqs)
    gt_seqs = Variable(gt_seqs)
    input_num = Variable(input_num)
    input_ppls = Variable(input_ppls)
    gt_bboxs = Variable(gt_bboxs)
    mask_bboxs = Variable(mask_bboxs)

    ####################################################################################
    # Build the Model
    ####################################################################################
    opt.vocab_size = dataset_val.vocab_size
    opt.detect_size = dataset_val.detect_size
    opt.seq_length = opt.seq_length
    opt.fg_size = dataset_val.fg_size
    opt.fg_mask = torch.from_numpy(dataset_val.fg_mask).byte()
    opt.glove_fg = torch.from_numpy(dataset_val.glove_fg).float()
    opt.glove_clss = torch.from_numpy(dataset_val.glove_clss).float()
    opt.st2towidx = torch.from_numpy(dataset_val.st2towidx).long()

    opt.itow = dataset_val.itow
    opt.itod = dataset_val.itod
    opt.ltow = dataset_val.ltow
    opt.itoc = dataset_val.itoc

    if opt.att_model == 'topdown':
        model = AttModel.TopDownModel(opt)
    elif opt.att_model == 'att2in2':
        model = AttModel.Att2in2Model(opt)

    if opt.decode_noc:
        model._reinit_word_weight(opt, dataset_val.ctoi, dataset_val.wtoi)

    if args.start_from != None:
        # opt.learning_rate = saved_model_opt.learning_rate
        print('Loading the model %s...' %(model_path))
        model.load_state_dict(torch.load(model_path))

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)

    if opt.cuda:
        model.cuda()

    lang_stats = demo(opt)
