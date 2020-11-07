#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize.treebank import TreebankWordDetokenizer

if torch.cuda.is_available():  
  device = "cuda:1"
else:
  device = "cpu"


def evaluate(val_loader, model, criterion, txt_field, device='cpu'):
  model.eval()
  min_val_loss = float('inf')
  epoch_loss = 0

  with torch.no_grad():
    for i, (imgs, caps, caplens, all_captions, all_caplens) in enumerate(val_loader):
      # Forward propagation
      output, target, lens = model(imgs.to(device), caps.to(device), caplens.to(device))
      outseq = output
      targetseq = target[:lens[0]]

      # Remove init_token
      # targetseq = targetseq[1:]
      # outseq = outseq[:, 1:].squeeze()
      loss = criterion(outseq, torch.LongTensor(targetseq))
      #     val_loss += loss.item()

      # References
      references = []
      for j in range(all_captions.shape[0]):
        temp = all_captions[j].tolist()
        img_caption = list(txt_field.vocab.itos[w] for w in temp if w not in
                           [txt_field.vocab.stoi[txt_field.init_token],
                            txt_field.vocab.stoi[txt_field.pad_token]
                            # txt_field.vocab.stoi[txt_field.end_token]
                            ])  # remove init and pad_tokens
        img_caption = TreebankWordDetokenizer().detokenize(img_caption)
        references.append(img_caption)

      # Hypotheses
      hypotheses = []
      _, predictions = torch.max(outseq, dim=1)
      predictions = predictions.tolist()
      
      temp_predictions = predictions
      pred_caption = list(txt_field.vocab.itos[w] for w in temp_predictions if w not in
                           [txt_field.vocab.stoi[txt_field.init_token],
                            txt_field.vocab.stoi[txt_field.pad_token]])
      pred_caption = TreebankWordDetokenizer().detokenize(pred_caption)
      hypotheses.append(pred_caption)
      print(pred_caption)
      print(references[0])
        
      for j in range(4):
        hypotheses.extend([pred_caption])

      assert len(references) == len(hypotheses)

      # Calculate BLEU scores
      bleu_1 = corpus_bleu(references, hypotheses, weights=[1,0,0,0])
      bleu_2 = corpus_bleu(references, hypotheses, weights=[0.5,0.5,0,0])
      bleu_3 = corpus_bleu(references, hypotheses, weights=[0.33,0.33,0.33,0])
      bleu_4 = corpus_bleu(references, hypotheses, weights=[0.25,0.25,0.25,0.25])

      print('Individual 1-gram: %.3f', bleu_1)
      print('Individual 2-gram: %.3f', bleu_2)
      print('Individual 3-gram: %.3f', bleu_3)
      print('Individual 4-gram: %.3f', bleu_4)
      
  return

def train(train_loader, val_loader, txt_field, model, optimizer, criterion, num_epoch, device):
    model.train()
    print_freq = 1
    acc = []
    start = time.time()
    train_loss = 0
    loss_history = []

    for epoch in range(num_epoch):
      train_loss = 0
      # Batchesize = 1
      for i, (imgs, caps, caplens) in enumerate(train_loader):
          if caplens[0] == 0: continue
          # Forward propagation
          output, target, lens = model(imgs, caps, caplens)

          # outseq = torch.argmax(output, dim=-1)
          outseq = output
          targetseq = target[1:lens[0]+1]

          # Remove init_token
          # targetseq = targetseq[1:]
          # outseq = outseq[:, 1:].squeeze().to(device)
          if len(outseq.shape)==1:
              outseq = outseq.unsqueeze(0)
          # print(outseq.shape)
          # print(targetseq.shape)
          loss = criterion(outseq, targetseq)
          train_loss += loss.item()
          print(loss.item())
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
          optimizer.step()

      # val_loss = evaluate(val_loader, model, criterion, txt_field, device)
      with open('history.txt','a') as f:
                f.write('Epoch {0:<3}'.format(epoch)+'\n')
                f.write('Loss {0:10.6f}'.format(train_loss)+'\n')

      loss_history.append(train_loss)

      if epoch%print_freq == 0:
        print('Epoch {0:<3}'.format(epoch))
        print('Loss {0:10.6f}'.format(train_loss))
        #print('Validation loss {0:10.6f}'.format(val_loss))

        PATH = "model_"+str(epoch)+".pt"
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        # 'val_loss': val_loss,
        }, PATH)

    return loss_history
            
