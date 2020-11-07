#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import  h5py
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

transform = transforms.Compose([
        transforms.Resize(224),
        torch.FloatTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, data_name, transform=None):
       
        self.split = split

        # assert self.split in {'train', 'val', 'test'}

        # Open hdf5 file where images and tokens are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '/' + data_name), 'r')
        self.imgs = self.h['images']
        print(self.imgs.shape)
        self.captions = np.int32(self.h['tokens'])
        self.caplens = self.h['sentence_length']
        # Captions per image
        self.cpi = 5

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.caplens)

    def __len__(self):

        return self.dataset_size

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        for ch in range(3):
            img[ch,:,:] = (img[ch,:,:] - mean[ch])/std[ch]
        
        # print(self.imgs[i // self.cpi].shape)
        # img = Image.fromarray(np.uint8(self.imgs[i // self.cpi].transpose(1,2,0))).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        caption = torch.tensor(self.captions[i,:], dtype=torch.long)

        caplen = torch.tensor([self.caplens[i]], dtype=torch.long)
       
        if self.split is 'train':
            return img, caption, caplen
        else:
            # For validation or testing, also return all 'captions_per_image' captions to find BLEU score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            all_caplen = torch.tensor(
                [self.caplens[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)]],
                dtype=torch.long)
            return img, caption, caplen, all_captions, all_caplen

   