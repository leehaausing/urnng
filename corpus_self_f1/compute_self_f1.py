#!/usr/bin/env python3


'''
to run this file use code:
python get_binary_tree.py --test_file data/ptb-test.pkl --model_file /models/urnng_1010.pt --gpu 0 --seed 1010

'''
import sys
import os

import argparse
#import json
import random
#import shutil
#import copy

#import torch
#from torch import cuda
#import torch.nn as nn
#from torch.autograd import Variable
#from torch.nn.parameter import Parameter

#import torch.nn.functional as F
import numpy as np
#import time
#import logging
#from data import Dataset
#from models import RNNG
from utils import *
import pickle

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--test_file', default='data/ptb-test.pkl')
parser.add_argument('--model_file', default='')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=str)


def main(args):
    np.random.seed(3454)
    #torch.manual_seed(3454)
    #seed = args.seed

    seed_list = [1000, 1234, 1450, 1818, 2019, 2345, 2828, 3434, 3435, 9292]
    #seed_list = [1000, 1450, 1818, 2828, 3435, 9292, 1010]
    comb = []
    # seed1 is the gold answer, but actually does not matter because F1 is symmetric
    for i, seed1 in enumerate(seed_list):
        for j, seed2 in enumerate(seed_list[(i + 1):]):
            print("Compare {} and {}".format(seed1, seed2))
            corpus_f1 = [0, 0, 0]
            span_list1 = pickle.load(open('urnng_model/spans/urnng_' + str(seed1) + '_spans.p', "rb"))
            span_list2 = pickle.load(open('urnng_model/spans/urnng_' + str(seed2) + '_spans.p', "rb"))
            for i in range(len(span_list1)):
                span1, span2 = span_list1[i], span_list2[i]
                tp, fp, fn = get_stats(span2, span1)
                corpus_f1[0] += tp
                corpus_f1[1] += fp
                corpus_f1[2] += fn

            tp, fp, fn = corpus_f1
            prec = tp / (tp + fp)
            recall = tp / (tp + fn)
            corpus_f1 = 2 * prec * recall / (prec + recall) * 100 if prec + recall > 0 else 0.
            print("Corpus F1 = ", corpus_f1)
            with open('self_f1.txt', 'a') as f:
                f.write("{} {} {}".format(seed1, seed2, corpus_f1))
                f.write("\n")

    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    
    
