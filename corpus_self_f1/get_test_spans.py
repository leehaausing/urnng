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
import pickle
parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--test_file', default='data/ptb-test.pkl')
parser.add_argument('--model_folder', type = str, default='urnng_model', help = "directory to store the saved checkpoints")
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
#parser.add_argument('--seed', default=3435, type=str)

#seed_list = [1000, 1234, 1450,1818, 2019, 2345, 2828, 3434, 3435,9292]
seed_list = [1234, 2019, 2345,  3434]

def main(args):
    np.random.seed(3454)
    torch.manual_seed(3454)
    data = Dataset(args.test_file)
    for seed in seed_list:
        print("Dealing with seed ", seed)
        model_file = args.model_folder + '/urnng_' + str(seed) + '.pt'
        try:
            checkpoint = torch.load(model_file)
            model = checkpoint['model']
            #print("model architecture")
            #print(model)
            cuda.set_device(args.gpu)
            model.cuda()
            model.eval()
            span_list = []

            with torch.no_grad():
                for i in range(len(data)):
                    sents, length, batch_size, gold_actions, gold_spans, gold_binary_trees, other_data = data[i]
                    if i % 10 == 0 and i >= 10:
                        print('{} / 189'.format(i))
                    if length == 1:
                        continue
                    sents = sents.cuda()
                    ll_word, ll_action_p, ll_action_q, all_actions, q_entropy = model(sents, samples=8, has_eos=True)
                    actions = all_actions[:, 0].long().cpu()
                    for b in range(batch_size):
                        action = list(actions[b].numpy())
                        span_b = get_spans(action[:-2])
                        span_b_set = set(span_b[:-1])
                        span_list.append(span_b_set)

            print('Saving to pickle ...')
            pickle.dump(span_list, open('urnng_' + str(seed) + '_spans.p', "wb"))
        except (AssertionError, AttributeError):
            print("Something goes wrong with seed ", seed)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

"""
    seed = args.seed
    data = Dataset(args.test_file)  
    checkpoint = torch.load(args.model_file)
    model = checkpoint['model']
    print("model architecture")
    print(model)
    cuda.set_device(args.gpu)
    model.cuda()
    model.eval()
    span_list = []
    
    with torch.no_grad():
        for i in range(len(data)):
            sents, length, batch_size, gold_actions, gold_spans, gold_binary_trees, other_data = data[i]
            if i % 10 == 0 and i >= 10:
                print('{} / 189'.format(i))
            if length == 1:
                continue
            sents = sents.cuda()
            ll_word, ll_action_p, ll_action_q, all_actions, q_entropy = model(sents, samples=8, has_eos = True)
            actions = all_actions[:, 0].long().cpu()
            for b in range(batch_size):
                action = list(actions[b].numpy())
                span_b = get_spans(action[:-2])
                span_b_set = set(span_b[:-1])
                span_list.append(span_b_set)
    
    print('Saving to pickle ...')
    pickle.dump(span_list, open('urnng_'+seed+'_spans.p', "wb"))

    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
"""
    