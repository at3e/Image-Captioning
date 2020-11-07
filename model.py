#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:30:15 2020

@author: at3ee
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import matplotlib.pyplot as plt

class EmbeddingLayer(nn.Module):
    def __init__(self, device, weight_matrix, non_trainable=True):
        super(EmbeddingLayer, self).__init__()
        self.num_embeddings, self.embedding_dim = weight_matrix.shape

        self.emb_layer = nn.Embedding(torch.tensor(self.num_embeddings), torch.tensor(self.embedding_dim))
        #, padding_idx=torch.LongTensor(self.pad_idx).to(device))
        self.emb_layer.weights = torch.tensor(weight_matrix, dtype=torch.float).to(device)
        if non_trainable:
            self.emb_layer.weight.requires_grad = False 
            
        self.device = device
            
    def forward(self,x):
        return self.emb_layer(x.to(self.device))


class Encoder(nn.Module):
    def __init__(self, device, nVopt, enc_img_size=3):
      super(Encoder, self).__init__()
      self.encoder_dim = nVopt['encoder_dim']
      self.D = nVopt['nV_ndescriptors']
      self.K = nVopt['nV_nclusters']
      self.alpha = nVopt['nV_alpha']
      self.fine_tune = nVopt['fine_tune']
      resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet
      # Remove linear and pool layers 
      modules = list(resnet.children())[:-2]
      self.resnet = nn.Sequential(*modules)
      self.linear = nn.Linear(self.encoder_dim, self.D)
      self.nVLAD = NetVLAD( self.K, self.D, self.alpha)
      self.device = device

      if self.fine_tune:
        for p in self.resnet.parameters():
          p.requires_grad = True
      # only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
          for p in c.parameters():
              p.requires_grad = self.fine_tune

    def forward(self, images):
      x = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
      x = x.permute(0, 2, 3, 1)  # (batch_size, image_size/32, image_size/32, 2048)
      x = self.linear(x)  # (batch_size, image_size/32, image_size/32, 512)
      x = x.permute(0, 3, 1, 2)
      x = self.nVLAD(x)

      return x.to(self.device)


class NetVLAD(nn.Module):
    def __init__(self, num_clusters, dim, alpha, if_normalize=True):
      super(NetVLAD, self).__init__()
      self.K = num_clusters
      self.D = dim
      self.alpha = alpha
      self.normalize_input = if_normalize
      self.CL = nn.Conv2d(self.D, self.K, kernel_size=(1, 1), bias=True)
      self.C = nn.Parameter(torch.rand(self.K, self.D))
      # Initialize parameters
      self.CL.weight = nn.Parameter(
          (2.0 * self.alpha * self.C).unsqueeze(-1).unsqueeze(-1)
          ) #DxKx1x1
      self.CL.bias = nn.Parameter(
          - self.alpha * self.C.norm(dim=1)
          )
      

    def forward(self, x):
      """
         param x: Input feature map of dimension [batch_size x channels x W x H]
      """
      
      # reshape input
      # N = W x H
      # D = channels
      x = x.flatten(start_dim=2).permute(2,1,0).unsqueeze(-1)
      
      # input feature is of shape [N x D x batch_size x 1]
      N, D = x.shape[:2] # N = dimension of input features, D = descriptors
      
      if self.normalize_input:
          x = F.normalize(x, p=2, dim=1)  # across descriptor dimension
          
      # soft-assignment
      S = self.CL(x).view(N, self.K, -1) # N x K x batch_size
      S = F.softmax(S, dim=1)
      
      x_flat = x.view(N, D, -1) # N x D x batch_size
      
      # VLAD core
      R = x_flat.expand(self.K, -1, -1, -1).permute(1, 0, 2, 3) \
          - self.C.expand(x_flat.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
      x = R*S.unsqueeze(2) # N x K x D x 1
      x = x.sum(dim=0) # K x D x 1  

      # normalization
      x = F.normalize(x, p=2, dim=0) # intra-normalize
      x = F.normalize(x, p=2, dim=1)  # L2 normalize
      x = x.permute(2,0,1) # batch_size x K x D

      return x
  

class DecoderLSTMLayer(nn.Module):
  def __init__(self, device, decoder_opt, dropout=0.2, phase='train'):
    super(DecoderLSTMLayer, self).__init__()
    self.n_layers = decoder_opt['n_layers']
    self.encoder_outdim = decoder_opt['encoder_dim']
    self.num_clusters = decoder_opt['nV_nclusters']
    self.output_dim = decoder_opt['vocab_size']
    self.hid_dim = decoder_opt['hid_dim']
    self.emb_weight_matrix = decoder_opt['emb_weights']
    self.emb_dim = self.emb_weight_matrix.shape[1]
    self.dropout = dropout    
    self.scale = self.scale = torch.sqrt(torch.FloatTensor([self.hid_dim]))

    self.emb_layer = EmbeddingLayer(device, self.emb_weight_matrix,
                                                      non_trainable=True)
    self.LSTM_layer = nn.LSTM(self.emb_dim+(self.encoder_outdim*self.num_clusters),
                                self.hid_dim, bias=True) # decoding RNNcell
    self.fc_out = nn.Linear(self.hid_dim, self.output_dim)    
    self.dropout = nn.Dropout(dropout)
    self.sigmoid = nn.Sigmoid()
    self.norm = nn.InstanceNorm1d(self.hid_dim)
    self.phase = phase
    self.device = device

  def init_hidden_state(self, encoder_out):
    mean_enc = encoder_out.mean(dim=-2)
    h = self.init_h(mean_enc)  # (batch_size, decoder_dim)
    c = self.init_c(mean_enc)
    return h.to(self.device), c.to(self.device)

  
  def forward(self, captions, encoder_out, cap_lens):
    batch_size = len(cap_lens)
    N = encoder_out.shape[1]
  
    # Embedding
    emb = self.emb_layer(captions[1:cap_lens+1]).unsqueeze(1)  # (batch_size, max_caption_length, embed_dim)
    encoder_out = encoder_out.view(1, -1).unsqueeze(0)
    enc_out_copy = [encoder_out for _ in range(cap_lens)]
    enc_out_copy = torch.cat(enc_out_copy).view(cap_lens, 1, -1)
    inputseq = torch.cat((emb,enc_out_copy), -1)
    # initialize the hidden state.
    hidden = (torch.randn(1, 1, self.hid_dim), torch.randn(1, 1, self.hid_dim))
    out, hid = self.LSTM_layer(inputseq, hidden)
    predictions = self.fc_out(out)
    predictions = F.softmax(predictions, dim = -1)
    
    return predictions.squeeze().to(self.device), captions.to(self.device), cap_lens

# Define image captioning
class ImageCaptioningModel(nn.Module):
    """
    Encoder-decoder architecture
    """
    def __init__(self, encoder_opt, decoder_opt, device, phase='train'):
      super(ImageCaptioningModel, self).__init__()
      self.encoder = Encoder(device, encoder_opt)
      self.decoder = DecoderLSTMLayer(device, decoder_opt, phase=phase)
      # self.decoder = DecoderLSTM(decoder_opt)

    def encode(self, x):
      if len(x.shape)<4:
            batch_size = 1
            x = x.unsqueeze(0)
      else:
            batch_size = x.size(0)

      out = self.encoder(x)
      # out = out.view(batch_size, -1)
      return out

    def decode(self, enc_out, tgt, tgt_lens, cell=None):
      batch_size = enc_out.size(0)
      pred, cap_sorted, dec_lens = self.decoder(tgt, enc_out, tgt_lens)
      return pred, cap_sorted, dec_lens


    def forward(self, src, tgt, tgt_lens):
      enc_out = self.encode(src)
      dec_out, cap_sorted, dec_lens = self.decode(enc_out=enc_out, 
                                                            tgt=tgt, tgt_lens=tgt_lens)

      return dec_out, cap_sorted, dec_lens