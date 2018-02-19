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

def build_dict(imgs, info, wtoi, wtod, dtoi, wtol, itod, params):
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
    bbox_ann = []
    bbox_idx = 0
    for sent in img:
        sent['bbox_idx'] = []
        for i, box in enumerate(sent['bbox']):
            sent['bbox_idx'].append(bbox_idx)
            bbox_ann.append({'bbox':box, 'label': dtoi[sent['clss'][i]], 'bbox_idx':bbox_idx})
            bbox_idx += 1      
    gt_bboxs = np.zeros((len(bbox_ann), 6))
    for i, bbox in enumerate(bbox_ann):
        gt_bboxs[i, :4] = bbox['bbox']
        gt_bboxs[i, 4] = bbox['label']
        gt_bboxs[i, 5] = bbox['bbox_idx']    

    if (params['split'] == info['images'][idx]['split']) or \
      (params['split'] == 'train' and info['images'][idx]['split'] == 'restval') or \
      (params['split'] == 'all'):
      #(params['split'] == 'val' and img['split'] == 'restval') or \
      ref_words = []
      ref_idxs = []

      captions = []
      for sent in img:
        sent['caption'] = sent['caption'] + ['<eos>']
        sent['caption'] = [_ if _ in wtoi else 'UNK' for _ in sent['caption']]
        captions.append(sent)

      det_indicator = get_det_word(gt_bboxs, captions, wtod, dtoi)
      
      ncap = len(captions) # number of captions available for this image
      for i, caption in enumerate(captions):
          tmp_tokens = []
          j = 0
          while j < len(caption['caption']):
            if det_indicator[i][j][0] != 0:
              tmp_tokens.append(vocab_size + det_indicator[i][j][2] * 2 + det_indicator[i][j][1]-1)
            else:
              tmp_tokens.append(wtoi[caption['caption'][j]])
            j += 1       
          ref_idxs.append(' '.join([str(int(_)) for _ in tmp_tokens]))
      # refs_words.append(ref_words)
      refs_idxs.append(ref_idxs)
      count_imgs += 1

  print('total imgs:', count_imgs)

  # ngram_words = compute_doc_freq(create_crefs(refs_words))
  ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
  return ngram_idxs, count_imgs


def get_det_word(gt_bboxs, captions, wtod, dtoi):
    
    # get the present category.
    pcats = []
    for i in range(gt_bboxs.shape[0]):
        pcats.append(gt_bboxs[i,5])
    # get the orginial form of the caption.
    indicator = []
    for i, sent in enumerate(captions):
        indicator.append([(0, 0, 0)]*len(sent['caption'])) # category class, binary class, fine-grain class.
        for j, bbox_idx in enumerate(sent['bbox_idx']):
            # if the bbox_idx is not filtered out.
            if bbox_idx in pcats:
                w_idx = sent['idx'][j]
                ng = sent['clss'][j]
                bn = (ng != sent['caption'][w_idx]) + 1
                fg = dtoi[ng]
                indicator[i][w_idx] = (wtod[sent['clss'][j]], bn, fg)
                
    return indicator

def main(params):

  info = json.load(open(params['dict_json'], 'r'))
  imgs = json.load(open(params['input_json'], 'r'))

  itow = info['ix_to_word']
  wtoi = {w:i for i,w in itow.items()}
  wtod = {w:i+1 for w,i in info['wtod'].items()} # word to detection
  # dtoi = {w:i+1 for i,w in enumerate(wtod.keys())} # detection to index
  dtoi = wtod
  wtol = info['wtol']
  itod = {i:w for w,i in dtoi.items()}

  # imgs = imgs['images']

  ngram_idxs, ref_len = build_dict(imgs, info, wtoi, wtod, dtoi, wtol, itod, params)

  # cPickle.dump({'document_frequency': ngram_words, 'ref_len': ref_len}, open(params['output_pkl']+'-words.p','w'), protocol=cPickle.HIGHEST_PROTOCOL)
  cPickle.dump({'document_frequency': ngram_idxs, 'ref_len': ref_len}, open(params['output_pkl']+'-idxs.p','w'), protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_json', default='data/flickr30k/cap_flickr30k.json', help='input json file to process into hdf5')
  parser.add_argument('--dict_json', default='data/flickr30k/dic_flickr30k.json', help='output json file')
  parser.add_argument('--output_pkl', default='data/flickr30k-train', help='output pickle file')
  parser.add_argument('--split', default='train', help='test, val, train, all')
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  main(params)
