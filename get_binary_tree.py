#!/usr/bin/env python3


'''
to run this file use code:
python get_binary_tree.py --test_file data/ptb-test.pkl --model_file /models/urnng_1010.pt --gpu 0 --seed 1010

'''
import sys
import os

import argparse
import json
import random
import shutil
import copy

import torch
from torch import cuda
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torch.nn.functional as F
import numpy as np
import time
import logging
from data import Dataset
from models import RNNG
from utils import *

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--test_file', default='data/ptb-test.pkl')
parser.add_argument('--model_file', default='')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=str)

def main(args):
    np.random.seed(3454)
    torch.manual_seed(3454)
    seed = args.seed
    data = Dataset(args.test_file)  
    checkpoint = torch.load(args.model_file)
    model = checkpoint['model']
    print("model architecture")
    print(model)
    cuda.set_device(args.gpu)
    model.cuda()
    model.eval()
    pred_list = []
    
    with torch.no_grad():
        for i in range(len(data)):
            sents, length, batch_size, gold_actions, gold_spans, gold_binary_trees, other_data = data[i]
            if i % 10 == 0 and i >= 10:
                print('{} / 189'.format(i))
            if length == 1:
                continue
            sents = sents.cuda()
            ll_word, ll_action_p, ll_action_q, all_actions, q_entropy = model(sents, samples=2,has_eos = True)
            actions = all_actions[:, 0].long().cpu()
            for bb in range(batch_size):
                action = list(actions[bb].numpy())
                sent_str = [data.idx2word[word_idx] for word_idx in list(sents[bb][1:-1].cpu().numpy())]
                pred = get_tree(action[:-2], sent_str)
                pred_list.append(pred)
    
    print('writing the txt file ...')
    with open('urnng_'+seed+'.txt','w') as f:
        for s in pred_list:
            f.write(s + '\n')
    

    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    
    