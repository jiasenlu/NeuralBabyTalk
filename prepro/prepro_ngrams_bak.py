"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""
import sys
import os
sys.path.append(os.getcwd())

import json
import argparse
from six.moves import cPickle
from collections import defaultdict
from pycocotools.coco import COCO
import numpy as np
import copy
import pdb

def precook(s, n=4, out=False):
  """
  Takes a string as input and returns an object that can be given to
  either cook_refs or cook_test. This is optional: cook_refs and cook_test
  can take string arguments as well.
  :param s: string : sentence to be converted into ngrams
  :param n: int    : number of ngrams for which representation is calculated
  :return: term frequency vector for occuring ngrams
  """
  words = s.split()
  counts = defaultdict(int)
  for k in xrange(1,n+1):
    for i in xrange(len(words)-k+1):
      ngram = tuple(words[i:i+k])
      counts[ngram] += 1
  return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def create_crefs(refs):
  crefs = []
  for ref in refs:
    # ref is a list of 5 captions
    crefs.append(cook_refs(ref))
  return crefs

def compute_doc_freq(crefs):
  '''
  Compute term frequency for reference data.
  This will be used to compute idf (inverse document frequency later)
  The term frequency is stored in the object
  :return: None
  '''
  document_frequency = defaultdict(float)
  for refs in crefs:
    # refs, k ref captions of one image
    for ngram in set([ngram for ref in refs for (ngram,count) in ref.iteritems()]):
      document_frequency[ngram] += 1
      # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
  return document_frequency

def build_dict(imgs, info, wtoi, wtod, dtoi, wtol, ctol, coco_det_train, coco_det_val, params):
  vocab_size = len(wtoi)+1
  seq_length = 16
  wtoi['<eos>'] = 0
  wtol['<eos>'] = '<eos>'
  count_imgs = 0

  refs_words = []
  refs_idxs = []
  for idx, img in enumerate(imgs):
    image_id = info['images'][idx]['id']
    # image_id = img['cocoid']
    file_path = info['images'][idx]['file_path'].split('/')[0]
    
    if file_path == 'train2014':
      coco = coco_det_train
    else:
      coco = coco_det_val
      
    bbox_ann_ids = coco.getAnnIds(imgIds=image_id)
    bbox_ann = [{'label': ctol[i['category_id']], 'bbox': i['bbox']} for i in coco.loadAnns(bbox_ann_ids)]

    if (params['split'] == info['images'][idx]['split']) or \
      (params['split'] == 'train' and info['images'][idx]['split'] == 'restval') or \
      (params['split'] == 'all'):
      #(params['split'] == 'val' and img['split'] == 'restval') or \
      ref_words = []
      ref_idxs = []

      captions = []
      for sent in img:
        captions.append(sent + ['<eos>'])
      det_indicator = get_det_word(bbox_ann, captions, wtoi, wtod, dtoi, wtol)
      
      ncap = len(captions) # number of captions available for this image
      for i, caption in enumerate(captions):
          tmp_tokens = []
          j = 0
          k = 0
          while j < len(caption):
              is_det = False
              for n in range(2, 0, -1):
                  if det_indicator[n][i][j][0] != 0:
                      tmp_tokens.append(vocab_size + det_indicator[n][i][j][2] * 2 + det_indicator[n][i][j][1])
                      is_det = True
                      j += n # skip the ngram.
                      break
              if is_det == False:
                  tmp_tokens.append(wtoi[caption[j]])
                  j += 1                
              k += 1
          ref_idxs.append(' '.join([str(int(_)) for _ in tmp_tokens]))
      # refs_words.append(ref_words)
      refs_idxs.append(ref_idxs)
      count_imgs += 1

  print('total imgs:', count_imgs)

  # ngram_words = compute_doc_freq(create_crefs(refs_words))
  ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
  return ngram_idxs, count_imgs


def get_det_word(bbox_ann, captions, wtoi, wtod, dtoi, wtol, ngram=2):
    
    # get the present category.
    pcats = [box['label'] for box in bbox_ann]

    # get the orginial form of the caption.
    indicator = []
    stem_caption = []
    for s in captions:
        tmp = []
        for w in s:
            tmp.append(wtol[w])
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

def main(params):

  det_train_path = 'data/coco/annotations/instances_train2014.json'
  det_val_path = 'data/coco/annotations/instances_val2014.json'

  coco_det_train = COCO(det_train_path)
  coco_det_val = COCO(det_val_path)

  info = json.load(open(params['dict_json'], 'r'))
  imgs = json.load(open(params['input_json'], 'r'))

  itow = info['ix_to_word']
  wtoi = {w:i for i,w in itow.items()}
  wtod = {w:i+1 for w,i in info['wtod'].items()} # word to detection
  dtoi = {w:i+1 for i,w in enumerate(wtod.keys())} # detection to index
  wtol = info['wtol']
  ctol = {c:i+1 for i, c in enumerate(coco_det_train.cats.keys())}

  # imgs = imgs['images']

  ngram_idxs, ref_len = build_dict(imgs, info, wtoi, wtod, dtoi, wtol, ctol, coco_det_train, coco_det_val, params)

  # cPickle.dump({'document_frequency': ngram_words, 'ref_len': ref_len}, open(params['output_pkl']+'-words.p','w'), protocol=cPickle.HIGHEST_PROTOCOL)
  cPickle.dump({'document_frequency': ngram_idxs, 'ref_len': ref_len}, open(params['output_pkl']+'-idxs.p','w'), protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_json', default='data/coco/cap_coco.json', help='input json file to process into hdf5')
  parser.add_argument('--dict_json', default='data/coco/dic_coco.json', help='output json file')
  parser.add_argument('--output_pkl', default='data/coco-train', help='output pickle file')
  parser.add_argument('--split', default='train', help='test, val, train, all')
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  main(params)
