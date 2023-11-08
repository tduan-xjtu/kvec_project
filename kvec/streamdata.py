#!/usr/bin/python3

import numpy as np
import pandas as pd
import torch

class StreamData(torch.utils.data.dataset.Dataset):
    def __init__(self, file_path):
        super(StreamData,self).__init__()
        self.data = np.load(file_path).reshape(-1,804)

        self.max_token_idx = int(self.data[:,:200].max())
        self.min_token_idx = int( self.data[:,:200].min())
        self.voc_len = self.max_token_idx - self.min_token_idx + 1

        self.labels = self.data[:,-2:]
        self.categories = list(set(self.labels.reshape(1,-1).tolist()[0]))
        self.num_classes = len(self.categories)

    def __getitem__(self, index):
        label = self.labels[index]
        label_id = np.asarray([self.categories.index(l) for l in label])
        #Debug print(self.categories)

        value = self.data[index,:200]
        key = self.data[index,200:400]
        burst = self.data[index,400:600]
        ret_pos = self.data[index,600:800]

        return  torch.from_numpy(value).long(), torch.from_numpy(key).long(),\
                torch.from_numpy(burst).long(),torch.from_numpy(ret_pos).long(),label_id

    def __len__(self):
        return self.data.shape[0]

