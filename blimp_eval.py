import sys
import os

import argparse
import json
import random
import shutil
import copy
import spacy
import pickle
from tqdm import tqdm

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

parser.add_argument('--test_file', default='data/ptb-test.pkl')
parser.add_argument('--model_file', default='')
parser.add_argument('--is_temp', default=2., type=float, help='divide scores by is_temp before CRF')
parser.add_argument('--samples', default=100, type=int, help='samples for IS calculation')
parser.add_argument('--count_eos_ppl', default=0, type=int, help='whether to count eos in val PPL')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int)
parser.add_argument('--eval_batch_size', default=16, type=int, help='batch size in evaluation')
parser.add_argument('--save_dir', default='/scratch/xl3119/urnng/blimp_results/sample.p', help='the saving directory of output data.')
parser.add_argument('--model_type', default='urnng', help='which model you are evaluating')

def read_model(model_dir):
    check_pt = torch.load(model_dir)
    return check_pt['model'], check_pt['word2idx'], check_pt['idx2word']

def read_json_file(data_dir):
    with open(data_dir, 'r', encoding="utf-8-sig") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data, data[0]['field'], data[0]['UID']

def tokenize(sent_good, sent_bad, vocab):
    nlp = spacy.load("en_core_web_sm")
    sent_tok_good =  [token.text for token in nlp(sent_good)]
    sent_tok_bad = [token.text for token in nlp(sent_bad)]
    sent_tok_good = [vocab[elem] if elem in vocab else vocab['<unk>'] for elem in sent_tok_good]
    sent_tok_bad = [vocab[elem] if elem in vocab else vocab['<unk>'] for elem in sent_tok_bad]
    return [vocab['<s>']]+sent_tok_good+[vocab['</s>']], [vocab['<s>']]+sent_tok_bad+[vocab['</s>']]

def rnnlm_compute_ll(sents, rnn_model):
    sents = torch.from_numpy(np.asarray(sents)).cuda()
    batch_size, length = sents.size()
    length -= 2
    cuda.set_device(0)
    rnn_model = rnn_model.cuda()
    rnn_model.eval()
    ll = rnn_model(sents)

    return ll.tolist()

def urnng_compute_ll(sents, model):
    sents = torch.from_numpy(np.asarray(sents)).cuda()
    batch_size, length = sents.size()
    length -= 2
    cuda.set_device(args.gpu)
    model = model.cuda()
    model.eval()
    num_sents = 0
    num_words = 0
    total_nll_recon = 0.
    total_nll_iwae = 0.
    samples_batch = 50
    S = args.samples // samples_batch
    samples = S*samples_batch
    if args.count_eos_ppl == 1:
        length += 1
    else:
        sents = sents[:, :-1]
    with torch.no_grad():
        ll_word_all2 = []
        ll_action_p_all2 = []
        ll_action_q_all2 = []
        for i in range(S):
            ll_word_all, ll_action_p_all, ll_action_q_all, actions_all, q_entropy = model(sents,
                                                        samples = samples_batch,
                                                        is_temp = args.is_temp,
                                                        has_eos = args.count_eos_ppl == 1)
            ll_word_all2.append(ll_word_all.detach().cpu())
            ll_action_p_all2.append(ll_action_p_all.detach().cpu())
            ll_action_q_all2.append(ll_action_q_all.detach().cpu())
        ll_word_all2 = torch.cat(ll_word_all2, 1)
        ll_action_p_all2 = torch.cat(ll_action_p_all2, 1)
        ll_action_q_all2 = torch.cat(ll_action_q_all2, 1)
        sample_ll = torch.zeros(batch_size, ll_word_all2.size(1))
        for j in range(sample_ll.size(1)):
            ll_word_j, ll_action_p_j, ll_action_q_j = ll_word_all2[:, j], ll_action_p_all2[:, j], ll_action_q_all2[:, j]
            sample_ll[:, j].copy_(ll_word_j + ll_action_p_j - ll_action_q_j)
        ll_iwae = model.logsumexp(sample_ll, 1) - np.log(samples)

        return ll_iwae.tolist()

def main(args):
    err = 0
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, vocab, idx2word = read_model(args.model_file)
    data, field, UID = read_json_file(args.test_file)

    #Filter out data that have <unk> tokens
    new_data = []
    for i, pair in enumerate(data):
        good_sent, bad_sent = pair["sentence_good"], pair["sentence_bad"]
        good_tok, bad_tok = tokenize(good_sent, bad_sent, vocab)
        if (vocab["<unk>"] not in good_tok) and (vocab["<unk>"] not in bad_tok):
            new_data.append((i,len(good_tok)-2,'good',good_tok))
            new_data.append((i,len(bad_tok)-2,'bad',bad_tok))

    #Sort data by length in ascending order
    #Get all lengths
    #Get number of sentences in all lengths
    sorted_new_data = sorted(new_data, key = lambda elem: elem[1])
    lengths = list(set([elem[1] for elem in new_data]))

    #Get all_negative_log_likelihoods
    all_ll = []
    for _len in lengths:
        ids = [elem[0] for elem in sorted_new_data if elem[1] == _len]
        all_sents = [elem[3] for elem in sorted_new_data if elem[1] == _len]
        num_sents = len(all_sents)
        sents_seg = [i for i in range(0,num_sents,args.eval_batch_size)]
        if sents_seg[-1] < num_sents:
            sents_seg.append(num_sents)
        for i in range(len(sents_seg)-1):
            sents = all_sents[sents_seg[i]:sents_seg[i+1]]
            if args.model_type == 'urnng':
                all_ll.extend(urnng_compute_ll(sents, model))
            elif args.model_type == 'rnnlm':
                all_ll.extend(rnnlm_compute_ll(sents, model))

    data_ll = [(sorted_new_data[i], all_ll[i]) for i in range(len(sorted_new_data))]
    sorted_data_ll = sorted(data_ll, key = lambda elem: elem[0][0])

    results = []
    for i, elem in enumerate(sorted_data_ll):
        if elem[0][0] not in [elem['PairID'] for elem in results]:
            results.append({'PairID':elem[0][0]})
        results[-1][elem[0][2]] = {}
        results[-1][elem[0][2]]['idxs'] = elem[0][3]
        results[-1][elem[0][2]]['sent'] = [idx2word[ele] for ele in elem[0][3][1:-1]]
        results[-1][elem[0][2]]['ll'] = elem[1]

    true_pred = 0
    num_pair = len(results)
    for elem in results:
        if elem['good']['ll'] > elem['bad']['ll']:
            true_pred += 1
    print("Field: {}, UID: {}".format(field, UID))
    print('Acc:{}'.format(true_pred / num_pair))

    #Dump data to the the output directory
    pickle.dump(results,open(args.save_dir, "wb"))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
