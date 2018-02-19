import json
import pdb
import numpy as np
import h5py

dataset = 'coco'

if dataset == 'coco':
    det_train = json.load(open('data/coco_noc/coco_detection_noc_train.json'))
    det_val = json.load(open('data/coco_noc/coco_detection_noc_val.json'))
    info = json.load(open('data/coco_noc/dic_coco.json'))

    det = []
    for img in det_train:
        img['split'] = 'train2014'
        det.append(img)

    for img in det_val:
        img['split'] = 'val2014'
        det.append(img)
elif dataset == 'flickr30k':
    det_file = json.load(open('data/flickr30k/flickr30k_detection.json'))
    info = json.load(open('data/flickr30k/dic_flickr30k.json'))
    det = []
    for img in det_file:
        det.append(img)

proposal_file = {}
for img in det:
    proposal_file[img['image_id']] = img

N = len(det)
dets_labels = np.zeros((N, 100, 6))
dets_num = np.zeros((N))
nms_num = np.zeros((N))

for idx, img in enumerate(info['images']):
    image_id = img['id']
    proposal = proposal_file[image_id]

    num_proposal = len(proposal['detection'])

    num_nms = proposal['num_boxes']
    proposals = np.zeros([num_proposal, 6])
    for i in range(num_proposal):
        proposals[i, :4] = proposal['detection'][i]['location']
        proposals[i, 4] = proposal['detection'][i]['label']
        proposals[i, 5] = proposal['detection'][i]['score']

    dets_labels[idx,:num_proposal] = proposals 
    dets_num[idx] = num_proposal
    nms_num[idx] = num_nms

if dataset == 'coco':
    f = h5py.File('coco_noc_detection.h5', "w")
elif dataset == 'flickr30k':
    f = h5py.File('flickr30k_detection.h5', "w")

f.create_dataset("dets_labels", data=dets_labels)
f.create_dataset("dets_num", data=dets_num)
f.create_dataset("nms_num", data=nms_num)
f.close()
