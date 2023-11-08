# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import torch
import torch.nn
import torch.optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import re
import os

from streamdata import StreamData
from modules import  MixHaltFormer,exponentialDecay
import torch.utils
import argparse

import logging
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Dataset hyperparameters
parser.add_argument("--dataset", type=str, default="Network Traffic",help="Dataset to load")

# Model hyperparameters
parser.add_argument("--embed_dim", type=int, default=128, help="Dimensions of the hidden vector")
parser.add_argument("--nhead", type=int, default=4, help="Number of attention head")
parser.add_argument("--nlayer", type=int, default=6, help="Number of attention layer")
parser.add_argument("--nclasses", type=int, default=12, help="Number of categories")
parser.add_argument("--nsubstream", type=int, default=2, help="Number of concurrent substreams")
parser.add_argument("--stream_len", type=int, default=256, help="Maximum length of substream")
parser.add_argument("--lam", type=float, default=0.0001, help="")
parser.add_argument("--bet", type=float, default=0.1, help="")

parser.add_argument("--rnn_cell", type=str, default='LSTM', help="Type of RNN to use in Fusion Layer")
parser.add_argument("--rnn_nhid", type=int, default=128, help="hidden layer deimensions of rnn")
parser.add_argument("--rnn_nlayers", type=int, default=1, help="layers of rnn")

# Training hyperparameters
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--nepochs", type=int, default=100, help="Number of epochs.")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate.")

# Dataset hyperparameters
parser.add_argument("--fold_num", type=str, default="fold_1", help="fold_1 -- fold_5")

args = parser.parse_args()

####Logger
def getLogger(logfile_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s")
    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    # FileHandler
    fHandler = logging.FileHandler(logfile_path, mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)

    return logger
####
logfile_save_path = '.\..'
logger = getLogger(logfile_save_path)
logger.info('Args : {}'.format(args))

config = {
    'early_compute_mode':'AVE_IN_BATC',
    'data_type':'MIX',
    'perform_compute_mode': 'SUBSTREAM',
    'mask_mode':'KVB_MASK'

}
logger.info('Config Information : {}'.format(config))

train_data_path = '.\..'
test_data_path = '.\..'
train_data = StreamData(train_data_path)
test_data = StreamData(test_data_path)

if train_data.categories == test_data.categories:
    logger.info('Real Lable to ID_Lable : {}'.format(train_data.categories))
else :
    logger.info('ERROR!!! : The label index between trainset and testset is different!!!')
    test_data.categories = train_data.categories
    logger.info('Transform the label keep same!')
    logger.info('Real Lable to ID_Lable : {}'.format(train_data.categories))

value2token_offset = abs(min(train_data.min_token_idx,test_data.min_token_idx))
max_token_idx = max(train_data.max_token_idx,test_data.max_token_idx)
Embedding_SIZE = max_token_idx + value2token_offset + 1

logger.info('Embedding_Size(vocabulary length) : {}'.format(Embedding_SIZE))

params = {'batch_size': args.batch_size, 'shuffle': True}

train_loader = DataLoader(train_data, **params)
test_loader = DataLoader(test_data, **params)

logger.info('Training Dataset : {}'.format(train_data_path))
logger.info('Testing Dataset : {}'.format(test_data_path))

def early_percentage(substream_lens,halt_lens,labels,category_num,early_compute_mode='AVE_IN_BATCH'):
    """
    :param stream_lens: shape: batch_size * sub-stream_nums
    :param halt_lens:
    :param category_num:
    :param early_compute_mode: 'AVE_IN_BATCH' or 'PER_SUBSTREAM'
    :return: a list
    """
    halt_percent_category = []

    if early_compute_mode == 'AVE_IN_BATCH':
        for i in range(category_num):
            substream_lens_cate ,halt_lens_cate = torch.zeros_like(substream_lens),torch.zeros_like(halt_lens)
            substream_lens_cate = torch.where(labels.unsqueeze(dim=2)==i,substream_lens,substream_lens_cate)
            halt_lens_cate = torch.where(labels.unsqueeze(dim=2)==i,halt_lens,halt_lens_cate)
            halt_percent = halt_lens_cate.sum() / substream_lens_cate.sum()
            halt_percent_category.append(halt_percent.item())

    else:
        halt_percent_all = halt_lens/substream_lens
        for i in range(category_num):
            category_halt = torch.zeros_like(halt_lens)
            category_halt = torch.where(labels.unsqueeze(dim=2)==i,halt_percent_all,category_halt)
            halt_percent = category_halt.sum()/(labels==0).sum()
            halt_percent_category.append(halt_percent.item())

    return halt_percent_category


def HM_cpmpute(acc, earliest):
    HM = 2 * (1 - earliest) * (acc) / ((1 - earliest) + acc)
    return np.round(HM,4)

def validation(val_data,data_mode='ORDER',compute_mode='CATEGORY'):
    """
    :param val_data:
    :param data_mode: 'ORDER' or 'MIX'
    :param compute_mode: 'CATEGORY' or 'SUBSTREAM' or 'BOTH'
    :return:
    """
    validation_predictions = []
    validation_labels = []
    early_substream = []

    model.eval()
    for i, (X, Key,burst,ret_pos, y) in enumerate(val_data):

        X = X.to(device)
        Key = Key.to(device)
        burst = burst.to(device)
        ret_pos = ret_pos.to(device)
        y = y.long().to(device)

        logits, halt_points, stream_len, substream_lens = model(X, Key, burst,ret_pos,value2token_offset,epoch)
        _, predictions = torch.max(torch.softmax(logits, dim=1), dim=1)

        substream_lens += 1.0
        halt_points += 1.0

        if compute_mode == 'CATEGORY':
            pass
        elif compute_mode == 'SUBSTREAM':
            halt_pre_substream = halt_points.mean() / substream_lens.mean()
            early_substream.append(halt_pre_substream.item())
        else:
            pass
        validation_predictions.append(predictions)
        y = torch.cat((y[:, 0], y[:, 1]), dim=0)
        validation_labels.append(y)

    validation_predictions = torch.stack(validation_predictions).cpu().numpy().reshape(-1, 1)
    validation_labels = torch.stack(validation_labels).cpu().numpy().reshape(-1, 1)

    accs = np.round(accuracy_score(validation_labels, validation_predictions), 3)
    precisions = np.round(precision_score(validation_labels, validation_predictions, average='macro'), 3)
    recalls = np.round(recall_score(validation_labels, validation_predictions, average='macro'), 3)
    f1s = np.round(f1_score(validation_labels, validation_predictions, average='macro'), 3)

    earliest = np.round(np.mean(early_substream), 3)
    performer_substream = [accs,earliest,precisions,recalls,f1s]

    return performer_substream

model = MixHaltFormer(d_model=args.embed_dim,pck_embedding_sizes=Embedding_SIZE,nhead=args.nhead,
                      num_encoder_layers=args.nlayer,num_classes=args.nclasses,num_substream=args.nsubstream,
                      dim_feedforward=2048, dropout=0.1, activation="relu",MASK_MODE=config['mask_mode'],
                      lam=args.lam,bet=args.bet,
                      rnn_cell=args.rnn_cell, rnn_nhid=args.rnn_nhid,rnn_nlayers=args.rnn_nlayers)

def weigth_init(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

model.apply(weigth_init)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info("using device : {}".format(device))
model.to(device)

#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

exponentials = exponentialDecay(args.nepochs) #epsilon-greedy is exponential decrease from 0.1 to 0 during training

####---- training ----####
training_loss = []
training_locations = []
training_predictions = []
results = []
result_save_path = '../result/kv_halt/{}/'.format(args.fold_num)+'c{}_K=5_mix_kvb_mask_l{}_b{}.csv'.format(args.nclasses,args.lam,args.bet)

for epoch in range(args.nepochs):
    model._epsilon = exponentials[epoch]
    loss_sum = 0
    model.train()
    for i, (X, Key,burst,ret_pos, y) in enumerate(train_loader):

        X = X.to(device)
        Key = Key.to(device)
        burst = burst.to(device)
        ret_pos = ret_pos.to(device)
        y = y.long().to(device)
        # --- Forward pass ---
        logits, _ , _ , _ = model(X, Key, burst,ret_pos,value2token_offset,epoch)
        _, predictions = torch.max(torch.softmax(logits, dim=1), dim=1)

        # --- Compute gradients and update weights ---
        optimizer.zero_grad()
        loss, loss_c, loss_r, loss_b,wait_penalty = model.computeLoss(logits, y)

        loss.backward()
        loss_sum += loss.item()
        optimizer.step()

        if (i+1) % 10 == 0:
            logger.info('Epoch [{}/{}], Batch [{}/{}], Total Loss: {:.4f} , CE Loss: {:.4f} , RL Loss: {:.4f}, '
                        'Baseline Loss: {:.4f}, Wait Penalty: {:.4f}'.format(epoch+1, args.nepochs, i+1, len(train_loader),
                                            loss.item(),loss_c.item(),loss_r.item(),loss_b.item(),wait_penalty.item()))

    performer_all_test_data = validation(val_data=test_loader,data_mode=config['data_type'],compute_mode=config['perform_compute_mode'])
    logger.info('Performer : {}'.format(performer_all_test_data))
    result = performer_all_test_data
    HM = HM_cpmpute(result[0],result[1])
    results.append([epoch] + result.tolist() + [HM])
    training_loss.append(np.round(loss_sum/len(train_loader), 3))
    #scheduler.step()

####----save model----####
results = pd.DataFrame(results)
results.to_csv(result_save_path,sep=',',header=False,index=False)
