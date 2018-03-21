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
plt.switch_backend('agg')
import json
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
    for step in range(1000):
        data = data_iter_val.next()
        img, iseq, gts_seq, num, proposals, bboxs, box_mask, img_id = data

        # if img_id[0] != 134688:
        #     continue

        # # for i in range(proposals.size(1)): print(opt.itoc[proposals[0][i][4]], i)

        # # list1 = [6, 10]
        # list1 = [0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9]
        # proposals = proposals[:,list1]
        # num[0,1] = len(list1)
        proposals = proposals[:,:max(int(max(num[:,1])),1),:]

        input_imgs.data.resize_(img.size()).copy_(img)
        input_seqs.data.resize_(iseq.size()).copy_(iseq)
        gt_seqs.data.resize_(gts_seq.size()).copy_(gts_seq)
        input_num.data.resize_(num.size()).copy_(num)
        input_ppls.data.resize_(proposals.size()).copy_(proposals)
        gt_bboxs.data.resize_(bboxs.size()).copy_(bboxs)
        mask_bboxs.data.resize_(box_mask.size()).copy_(box_mask)
        input_imgs.data.resize_(img.size()).copy_(img)

        eval_opt = {'sample_max':1, 'beam_size': opt.beam_size, 'inference_mode' : True, 'tag_size' : opt.cbs_tag_size}
        seq, bn_seq, fg_seq, _, _, _ = model._sample(input_imgs, input_ppls, input_num, eval_opt)

        sents, det_idx, det_word = utils.decode_sequence_det(dataset_val.itow, dataset_val.itod, dataset_val.ltow, dataset_val.itoc, dataset_val.wtod, \
                                                            seq, bn_seq, fg_seq, opt.vocab_size, opt)

        if opt.dataset == 'flickr30k':
            im2show = Image.open(os.path.join(opt.image_path, '%d.jpg' % img_id[0])).convert('RGB')
        else:

            if os.path.isfile(os.path.join(opt.image_path, 'val2014/COCO_val2014_%012d.jpg' % img_id[0])):
                im2show = Image.open(os.path.join(opt.image_path, 'val2014/COCO_val2014_%012d.jpg' % img_id[0])).convert('RGB')
            else:
                im2show = Image.open(os.path.join(opt.image_path, 'train2014/COCO_train2014_%012d.jpg' % img_id[0])).convert('RGB')

        w, h = im2show.size

        rest_idx = []
        for i in range(proposals[0].shape[0]):
            if i not in det_idx:
                rest_idx.append(i)


        if len(det_idx) > 0:
            # for visulization
            proposals = proposals[0].numpy()
            proposals[:,0] = proposals[:,0] * w / float(opt.image_crop_size)
            proposals[:,2] = proposals[:,2] * w / float(opt.image_crop_size)
            proposals[:,1] = proposals[:,1] * h / float(opt.image_crop_size)
            proposals[:,3] = proposals[:,3] * h / float(opt.image_crop_size)            

            cls_dets = proposals[det_idx]
            rest_dets = proposals[rest_idx]

        # fig = plt.figure()
        # fig = plt.figure(frameon=False)
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig = plt.figure(frameon=False)
        # fig.set_size_inches(5,5*h/w)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        a=fig.gca()
        a.set_frame_on(False)
        a.set_xticks([]); a.set_yticks([])
        plt.axis('off')
        plt.xlim(0,w); plt.ylim(h,0)
        # fig, ax = plt.subplots(1)

        # show other box in grey.

        plt.imshow(im2show)

        if len(rest_idx) > 0:
            for i in range(len(rest_dets)):
                ax = utils.vis_detections(ax, dataset_val.itoc[int(rest_dets[i,4])], rest_dets[i,:5], i, 1)

        if len(det_idx) > 0:
            for i in range(len(cls_dets)):
                ax = utils.vis_detections(ax, dataset_val.itoc[int(cls_dets[i,4])], cls_dets[i,:5], i, 0)

        # plt.axis('off')
        # plt.axis('tight')
        # plt.tight_layout()
        fig.savefig('visu/%d.jpg' %(img_id[0]), bbox_inches='tight', pad_inches=0, dpi=150)
        print(str(img_id[0]) + ': ' + sents[0])

        entry = {'image_id': img_id[0], 'caption': sents[0]}
        predictions.append(entry)

    return predictions
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
    parser.add_argument('--cbs', type=bool, default=False,
                    help='whether use constraint beam search.')
    parser.add_argument('--cbs_tag_size', type=int, default=3,
                    help='whether use constraint beam search.')
    parser.add_argument('--cbs_mode', type=str, default='all',
                    help='which cbs mode to use in the decoding stage. cbs_mode: all|unique|novel')
    parser.add_argument('--det_oracle', type=bool, default=False,
                    help='whether use oracle bounding box.')
    parser.add_argument('--cnn_backend', type=str, default='res101',
                    help='res101 or vgg16')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--beam_size', type=int, default=1)

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
            opt.cbs = args.cbs
            opt.cbs_tag_size = args.cbs_tag_size
            opt.cbs_mode = args.cbs_mode
            opt.det_oracle = args.det_oracle
            opt.cnn_backend = args.cnn_backend
            opt.data_path = args.data_path
            opt.beam_size = args.beam_size
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
    dataset_val = DataLoader(opt, split='test')
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1,
                                            shuffle=False, num_workers=0)

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

    pdb.set_trace()
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
        if os.path.isfile(os.path.join(args.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(args.start_from, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)

    if opt.cuda:
        model.cuda()

    predictions = demo(opt)

    print('saving...')
    json.dump(predictions, open('visu.json', 'w'))

