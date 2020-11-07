#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from preProcess import build_vocab
import spacy
import torch
import torch.nn as nn
from model import ImageCaptioningModel
from dataloaderUtils import CaptionDataset
from trainModel import train, evaluate
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

if torch.cuda.is_available():  
  device = "cuda:1"
else:
  device = "cpu"

capFilename = "capList.csv"
vec_file = "glove.6B.100d.txt"
spacy_en = spacy.load('en')
path = './GloVe/'
TXT_FIELD, VECTORS = build_vocab(path, vec_file, capFilename)
vocab_size = len(TXT_FIELD.vocab)
vec_dim = 100

# Create dataloader
data_folder = './'
train_dataset = CaptionDataset(data_folder, 'train', 'trainImages.h5')
val_dataset = CaptionDataset(data_folder, 'val', 'valImages.h5')
test_dataset = CaptionDataset(data_folder, 'test', 'testImages.h5')
ignore_tok = [TXT_FIELD.vocab.stoi[TXT_FIELD.init_token], 
              TXT_FIELD.vocab.stoi[TXT_FIELD.pad_token]]

# Create model
embed_weights = torch.load('embedLayer.pt')

encoder_spec = {'encoder_dim':2048, 'nV_ndescriptors':256, 'nV_nclusters':4,
                'nV_alpha':0.01, 'fine_tune':True}

decoder_spec = {'n_layers':1, 'encoder_dim':256, 'nV_nclusters': 4,
                'vocab_size':vocab_size, 'hid_dim':1024, 'emb_weights':embed_weights}
phase = 'test'
model = ImageCaptioningModel(encoder_spec, decoder_spec, device, phase)
# model.to(device)

# train model
LEARNING_RATE = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
num_epoch = 500
model = train(train_dataset, val_dataset, TXT_FIELD, model, optimizer, criterion, num_epoch, device)
