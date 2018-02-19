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
"""
to get the prepro file for neural baby talk. we need 2 additional dictionaries. 
wtol: word to lemma, find the orignial form of the word.
wtod: word to detection, find the detection label for the word.
"""
import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
import torchvision.models as models
from torch.autograd import Variable
import skimage.io
import pdb
from stanfordcorenlp import StanfordCoreNLP
from nltk.tokenize import word_tokenize

nlp = StanfordCoreNLP('../stanford-corenlp-full-2017-06-09', memory='8g')
props={'annotators': 'ssplit, tokenize, lemma','pipelineLanguage':'en', 'outputFormat':'json'}

def build_vocab(imgs, params):
  count_thr = params['word_count_threshold']

  # count up the number of words
  counts = {}
  for img in imgs:
    for sent in img['sentences']:
      # sent['tokens'] = word_tokenize(sent['raw'].lower())
      for w in sent['tokens']:
        counts[w] = counts.get(w, 0) + 1
  cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
  print('top words and their counts:')
  print('\n'.join(map(str,cw[:20])))

  # print some stats
  total_words = sum(counts.values())
  print('total words:', total_words)
  bad_words = [w for w,n in counts.items() if n <= count_thr]
  vocab = [w for w,n in counts.items() if n > count_thr]
  bad_count = sum(counts[w] for w in bad_words)
  print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
  print('number of words in vocab would be %d' % (len(vocab), ))
  print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

  # lets look at the distribution of lengths as well
  sent_lengths = {}
  for img in imgs:
    for sent in img['sentences']:
      txt = sent['tokens']
      nw = len(txt)
      sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
  max_len = max(sent_lengths.keys())
  print('max length sentence in raw data: ', max_len)
  print('sentence length distribution (count, number of words):')
  sum_len = sum(sent_lengths.values())
  for i in range(max_len+1):
    print('%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len))

  # lets now produce the final annotations
  if bad_count > 0:
    # additional special UNK token we will use below to map infrequent words to
    print('inserting the special UNK token')
    vocab.append('UNK')
  
  imgs_new = []
  for img in imgs:
    img['final_captions'] = []
    for sent in img['sentences']:
      txt = sent['tokens']
      caption = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
      img['final_captions'].append(caption)
    imgs_new.append(img['final_captions'])

  return vocab, imgs_new

def main(params):

  coco_class_all = []
  coco_class_name = open('data/coco/coco_class_name.txt', 'r')
  for line in coco_class_name:
      coco_class = line.rstrip("\n").split(', ')
      coco_class_all.append(coco_class)

  # word to detection label
  wtod = {}
  for i in range(len(coco_class_all)):
    for w in coco_class_all[i]:
      wtod[w] = i

  imgs = json.load(open(params['input_json'], 'r'))
  imgs = imgs['images']

  seed(123) # make reproducible
  
  # create the vocab
  vocab, imgs_new = build_vocab(imgs, params)
  itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
  wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

  wtol = {}
  for w in vocab:
      out = json.loads(nlp.annotate(w.encode('utf-8'), properties=props))
      lemma_w = out['sentences'][0]['tokens'][0]['lemma']
      wtol[w] = lemma_w

  if split == 'robust':
    split_path = 'data/coco_robust/split_robust_coco.json'
    split_file = json.load(open(split_path, 'r'))
  pdb.set_trace()
  # create output json file
  out = {}
  out['ix_to_word'] = itow # encode the (1-indexed) vocab
  out['wtod'] = wtod
  out['wtol'] = wtol
  out['images'] = []
  for i,img in enumerate(imgs):
    jimg = {}
    if img['split'] == 'val' or img['split'] == 'test':
      jimg['split'] = img['split']
    else:
      jimg['split'] = 'train' # put restrl into train.

    if 'filename' in img: jimg['file_path'] = os.path.join(img['filepath'], img['filename']) # copy it over, might need
    if 'cocoid' in img: jimg['id'] = img['cocoid'] # copy over & mantain an id, if present (e.g. coco ids, useful)
    out['images'].append(jimg)
  
  json.dump(out, open(params['outpu_dic_json'], 'w'))
  print('wrote ', params['outpu_dic_json'])

  json.dump(imgs_new, open(params['output_cap_json'], 'w'))
  print('wrote ', params['output_cap_json'])

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_json', default='data/coco/dataset_coco.json', help='input json file to process into hdf5')
  parser.add_argument('--split', default='robust', help='input json file to process into hdf5')

  parser.add_argument('--outpu_dic_json', default='data/coco/dic_coco.json', help='output json file')
  parser.add_argument('--output_cap_json', default='data/coco/cap_coco.json', help='output json file')

  # options
  parser.add_argument('--max_length', default=16, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))
  main(params)
