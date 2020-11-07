#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import csv
import json
import os
from torchtext import data, vocab
import spacy
import numpy as np
import string
import h5py
from PIL import Image

import torch
import torch.nn as nn

def tokenizer(s):
  translator=str.maketrans('','',string.punctuation)
  s = str(s).lower().translate(translator)
  return [w.text for w in spacy_en.tokenizer(s)]

def build_vocab(path, vec_file, cap_path):
  vector_en = vocab.Vectors(vec_file, path)
  txt_field = data.Field(sequential= True, use_vocab=True, init_token='<SOS>', eos_token='<EOS>',
                       lower=True, tokenize=tokenizer, pad_token='<pad>')
  field_names = [('Idx', None),
               ('Sentence', txt_field)]
  text = data.TabularDataset(path=cap_path, format='csv', fields=field_names, skip_header=True)
  txt_field.build_vocab(text.Sentence, max_size=100000, vectors=vector_en)

  return txt_field, vector_en

def encode_captions(file_list, txt_field, max_len=40):
  cpi = 5
  all_labels = np.zeros((len(file_list)*cpi, max_len))
  all_labels[:,:] = txt_field.vocab.stoi[txt_field.pad_token]
  label_len = np.zeros(len(file_list)*cpi)
  for i, entry in enumerate(file_list):
    capList = entry['captions']
    for j, cap in enumerate(capList):
      
      tokens = tokenizer(cap)
      labels = list(txt_field.vocab.stoi[t] for t in tokens)
      all_labels[i*cpi + j, 0] = txt_field.vocab.stoi[txt_field.init_token]
      all_labels[i*cpi + j, 1:len(labels)+1] = np.array(labels)
      all_labels[i*cpi + j, len(labels)+1] = txt_field.vocab.stoi[txt_field.eos_token]
      label_len[i*cpi + j] = len(labels)
  return all_labels, label_len

def embed_weights(vector_en, vocab, vector_dim):
    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, vector_dim))
    words_found = 0

    for i, word in enumerate(vocab.itos):
      try:
        weights_matrix[i] = vector_en.get_vecs_by_tokens(word)
      except KeyError:
        # random vectors for unknown words
        weights_matrix[i] = np.random.normal(scale=0.6, size=(vector_dim, ))

    return weights_matrix

img_dir = "Images"
cap_file = "33/captions.txt"
caption_dict = {}
all_captions = []
capFilename = "capList.csv"

with open(cap_file, "rb") as f:
  for l in f:
        line = l.decode().split('\t')
        text = re.sub('.\n',"",line[1])
        all_captions.append(line[1])
        tag = line[0].split('#')
        
        if tag[1]=='0':
            cList =[]
            cList.append(text)
            caption_dict[tag[0]] = cList
        else:
            temp = list(caption_dict[tag[0]])
            temp.append(text)
            caption_dict[tag[0]] = temp

with open(capFilename , "w") as capFile:
  cwriter = csv.writer(capFile)
  cwriter.writerow(('Idx', 'Sentence'))
  for i, entries in enumerate(all_captions) :
    cwriter.writerow((i, entries))


filenames = caption_dict.keys()
num_val = 2
num_test = 2
train_file_list, val_file_list, test_file_list = [], [], []
for i, k in enumerate(filenames):
  if i<num_test:
    Dict = {"file_path": k, "captions": caption_dict[k]}
    test_file_list.append(Dict)

  elif num_test<=i<num_val+num_test:
    Dict = {"file_path": k, "captions": caption_dict[k]}
    val_file_list.append(Dict)   

  else:
    Dict = {"file_path": k, "captions": caption_dict[k]}
    train_file_list.append(Dict)
    
with open("testFileList.json", "w") as f:
      json.dump(test_file_list, f)

with open("valFileList.json", "w") as f:
      json.dump(val_file_list, f)

with open("trainFileList.json", "w") as f:
      json.dump(train_file_list, f)


splits = ['train', 'val', 'test']
vec_file = "glove.6B.100d.txt"
spacy_en = spacy.load('en')

path = './GloVe/'
TXT_FIELD, VECTORS = build_vocab(path, vec_file, capFilename)
vocab_size = len(TXT_FIELD.vocab)

for s in splits:
  tar_dir = s
  if not os.path.isdir(tar_dir):
    os.mkdir(tar_dir)
  fname = s+"FileList.json"
  with open(fname, "r") as f:
    flist = json.load(f)
  N = len(flist)
  Labels, lenLabels = encode_captions(flist, TXT_FIELD)
  
  with h5py.File(tar_dir+'/'+s+'Images.h5', "w") as f:
    f.create_dataset("tokens", dtype='uint32', data=Labels)
    f.create_dataset("sentence_length", dtype='uint32', data=lenLabels)
    imageset = f.create_dataset("images", (N,3,224,224), dtype='uint8')
      
    for i,img in enumerate(flist):
      # load the image
      img_name = img_dir+ '/'+ img['file_path']
      try:
          I = Image.open(img_dir+ '/'+ img['file_path'])
      except:
          print(img['file_path'])
          continue
      newsize = (224, 224) 
      Ir = I.resize(newsize) # Resnet image size
      # handle grayscale images
      # if len(Ir.shape) == 2:
      #   Ir = Ir[:,:,np.newaxis]
      #   Ir = np.concatenate((Ir,Ir,Ir), axis=2)
      
      Ir = np.array(Ir).transpose(2,0,1)
      # write to h5
      imageset[i] = Ir
      
      if s == 'train' and i % 1000 == 0:      
        print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
f.close()

# create weight matrix for embedding layer
vec_dim = 100
embed_weights = embed_weights(VECTORS, TXT_FIELD.vocab, vec_dim)

# save embedding layer weights
torch.save(embed_weights, 'embedLayer.pt')
