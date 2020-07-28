import argparse
import re
import os
import matplotlib.pyplot as plt
import nltk
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable

import json
from torch.utils.data import Dataset, DataLoader
from test_data_loader import *
import numpy as np
from tqdm import tqdm
import pickle


# Test model
def build_tree(depth, sen):
    assert len(depth) == len(sen)

    if len(depth) == 1:
        parse_tree = sen[0]
    else:
        idx_max = numpy.argmax(depth)
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = build_tree(depth[:idx_max], sen[:idx_max])
            parse_tree.append(tree0)
        tree1 = sen[idx_max]
        if len(sen[idx_max + 1:]) > 0:
            tree2 = build_tree(depth[idx_max + 1:], sen[idx_max + 1:])
            tree1 = [tree1, tree2]
        if parse_tree == []:
            parse_tree = tree1
        else:
            parse_tree.append(tree1)
    return parse_tree


def test(model, dataloader, cuda, prt=False):

    model.eval()

    pred_tree_list = {}

    it = tqdm(range(len(dataloader)), ncols=0)
    data_iter = iter(dataloader)

    for niter in it:
        input_ids, length, idx, cand_ids, target = data_iter.next()
        if args.cuda:
            input_ids = input_ids.cuda()
            length = length.cuda()
            cand_ids = cand_ids.cuda()
            targets = target.cuda()

        hidden = model.init_hidden(1)
        hidd = model.init_hidden(1*input_ids.shape[1])
        hidd_cand = model.init_hidden(1*4)
        with torch.no_grad():
            output, result_prob, hidden, rnn_hs, dropped_rnn_hs, cand_emb = model(input_ids, cand_ids, hidden, hidd, hidd_cand)

        distance = model.distance[0].squeeze().data.cpu().numpy()[:, :length]
        
        for gates in [
            # distance[0],
            distance[1],
            # distance[2],
            # distance.mean(axis=0)
        ]:
            depth = gates
            parse_tree = build_tree(depth, list(np.arange(1., distance[1].shape[0]+1)))

            # print(json.dumps(parse_tree, indent=4))

            pred_tree_list[idx] = parse_tree


    return pred_tree_list

if __name__ == '__main__':
    marks = [' ', '-', '=']

    numpy.set_printoptions(precision=2, suppress=True, linewidth=5000)

    parser = argparse.ArgumentParser(description='PyTorch hierarchical ON-LSTM Language Model')

    # Model parameters.
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='location of the data corpus')
    parser.add_argument('--checkpoint', type=str, default='../model/ckpt.pt',
                        help='model checkpoint to use')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    args = parser.parse_args()
    args.bptt = 70

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    # Load model
    with open(args.checkpoint, 'rb') as f:
        model, _, _ = torch.load(f)
        torch.cuda.manual_seed(args.seed)
        model.cpu()
        if args.cuda:
            model.cuda()

    train = get_loader(args.data_dir, phase='train', batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    val = get_loader(args.data_dir, phase='test', batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    dataloader = [val, train]

    for i, loader in enumerate(dataloader):
        pred_tree_list = test(model, loader, args.cuda, prt=True)
        pickle.dump(pred_tree_list, open(os.path.join(args.data_dir, 'pred_tree_list_'+str(i)+'.pkl'), 'wb'))

