# import _init_paths
import copy
import json
import operator
from random import seed, shuffle

import numpy as np
from six.moves import xrange

from pycocotools.coco import COCO


def get_det_word(bbox_ann, captions, wtoi, wtod, dtoi, wtol, ngram=2):

    # get the present category.
    pcats = [box['label'] for box in bbox_ann]

    # get the orginial form of the caption.
    indicator = []
    stem_caption = []
    for s in captions:
        tmp = []
        for w in s:
            if w in wtol:
                tmp.append(wtol[w])
            else:
                tmp.append(w)

        stem_caption.append(tmp)
        indicator.append([(0, 0, 0)]*len(s)) # category class, binary class, fine-grain class.

    ngram_indicator = {i+1:copy.deepcopy(indicator) for i in range(ngram)}
    # get the 2 gram of the caption.
    for n in range(ngram,0,-1):
        for i, s in enumerate(stem_caption):
            for j in xrange(len(s)-n+1):
                ng = ' '.join(s[j:j+n])
                # if the n-gram exist in word_to_detection dictionary.
                if ng in wtod and indicator[i][j][0] == 0 and wtod[ng] in pcats: # make sure that larger gram not overwright with lower gram.
                    bn = (ng != ' '.join(captions[i][j:j+n])) + 1
                    fg = dtoi[ng]
                    ngram_indicator[n][i][j] = (wtod[ng], bn, fg)
                    indicator[i][j:j+n] = [(wtod[ng], bn, fg)] * n

    return ngram_indicator

def get_stats(imgs, wtoi, wtod, dtoi, wtol, ctol, coco_det_train, coco_det_val):

    train_matrix = np.zeros((len(wtod),len(wtod)))
    test_matrix = np.zeros((len(wtod),len(wtod)))
    test_num = 0
    coco_stats = []

    for idx, img in enumerate(imgs):

        image_id = info['images'][idx]['id']
        file_path = info['images'][idx]['file_path'].split('/')[0]

        if file_path == 'train2014':
            coco = coco_det_train
        else:
            coco = coco_det_val
        bbox_ann_ids = coco.getAnnIds(imgIds=image_id)
        bbox_ann = [{'label': ctol[i['category_id']], 'bbox': i['bbox']} for i in coco.loadAnns(bbox_ann_ids)]
        captions = []
        for sent in img['sentences']:
            captions.append(sent['tokens'])
        det_indicator = get_det_word(bbox_ann, captions, wtoi, wtod, dtoi, wtol)

        present_clss = []

        for i, caption in enumerate(captions):
            for j in range(len(caption)):
                for n in range(2, 0, -1):
                    if det_indicator[n][i][j][0] != 0 and det_indicator[n][i][j][0] not in present_clss:
                        present_clss.append(det_indicator[n][i][j][0])
        coco_stats.append({'pclss':present_clss, 'image_id':image_id})

    return coco_stats

imgs = json.load(open('data/robust_coco_creation/dataset_coco.json', 'r'))

det_train_path = 'data/robust_coco_creation/annotations/instances_train2014.json'
det_val_path = 'data/robust_coco_creation/annotations/instances_val2014.json'

coco_det_train = COCO(det_train_path)
coco_det_val = COCO(det_val_path)

info = json.load(open('data/robust_coco_creation/dic_coco.json', 'r'))
itow = info['ix_to_word']
wtoi = {w:i for i,w in itow.items()}
wtod = {w:i+1 for w,i in info['wtod'].items()} # word to detection
dtoi = {w:i+1 for i,w in enumerate(wtod.keys())} # detection to index
wtol = info['wtol']
ctol = {c:i+1 for i, c in enumerate(coco_det_train.cats.keys())}
imgs = imgs['images']
coco_stats = get_stats(imgs, wtoi, wtod, dtoi, wtol, ctol, coco_det_train, coco_det_val)
class_total = np.zeros(80)
# get the sum for each category.
for img in coco_stats:
    img['pclss'] = [i-1 for i in img['pclss']]
    for idx in img['pclss']:
        class_total[idx] += 1

json.dump(coco_stats, open('coco_obj_stats.json', 'w'))
pair_list = {}
for img in coco_stats:
    for i in range(len(img['pclss'])):
        for j in range(len(img['pclss'])):
            if i != j:
                idx_i = img['pclss'][i]
                idx_j = img['pclss'][j]
                if idx_i < idx_j:
                    idx_ij = (idx_i, idx_j)
                else:
                    idx_ij = (idx_j, idx_i)
                if idx_ij not in pair_list:
                    pair_list[idx_ij] = 0
                else:
                    pair_list[idx_ij] += 1

pair_list_sort = sorted(pair_list.items(), key=operator.itemgetter(1))

pair_list = []
for pair in pair_list_sort:
    pair_list.append([pair[0][0], pair[0][1], pair[1]])

# for each pair, go throughall the images
testing_total = np.zeros(80)
test_pair = []
count = 0
test_img_num = 0
for pair in pair_list:
    tmp_num = 0
    testing_total_copy = copy.deepcopy(testing_total)
    for img in coco_stats:
        if pair[0] in img['pclss'] and pair[1] in img['pclss']:
            # also accumulate other class.
            for idx in img['pclss']:
                testing_total_copy[idx] += 1
            tmp_num += 1

    # if the testing data exceed half of the total data, don't count this pair.
    drop_flag = False
    for i in range(80):
        if testing_total_copy[i] > (class_total[i] / 2):
            drop_flag = True
            print("drop pair " + str(pair[0]) + '_' + str(pair[1]))
            break

    if drop_flag == False:
        test_pair.append(pair)
        testing_total = copy.deepcopy(testing_total_copy)
        test_img_num += tmp_num

    count += 1
    print(count, test_img_num)
    if test_img_num > 15000:
        break

print('saving the test pair list....')
json.dump(test_pair, open('test_pair_list.json', 'w'))

test_pair_dic = {}
for pair in test_pair:
    test_pair_dic[str(pair[0])+'_'+str(pair[1])] = 0

train_img_id = []
test_img_id = []
for img in coco_stats:
    present_clss = img['pclss']

    # generate the pair.
    tmp  = []
    for i in range(len(present_clss)):
        for j in range(len(present_clss)):
            if i != j:
                tmp.append(str(present_clss[i]) + '_' + str(present_clss[j]))

    test_flag = False
    for i in tmp:
        if i in test_pair_dic:
            test_flag = True
    if test_flag == True:
        test_img_id.append({'img_id': img['image_id']})
    else:
        train_img_id.append({'img_id': img['image_id']})

seed(123) # make reproducible
shuffle(test_img_id) # shuffle the order

num_val = int(0.3 * len(test_img_id))

train_id = train_img_id
val_id = test_img_id[:num_val]
test_id = test_img_id[num_val:]

print("train, val, test", len(train_id), len(val_id), len(test_id))
robust_split = {'train_id':train_id, 'val_id':val_id, 'test_id':test_id}
json.dump(robust_split, open('split_robust_coco.json', 'w'))
