from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle
import random

class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = ['<pad>', '<unk>']
        # self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        # self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, item):
        if item in self.word2idx:
            return self.word2idx[item]
        else:
            return self.word2idx['<unk>']

class RecipeDataset(Dataset):

    def __init__(self, root_dir='../data', phase='train'):

        dict_file_name = os.path.join(root_dir, 'dict_recipe.pkl')
        self.tokenizer = pickle.load(open(dict_file_name, 'rb'))
        self.max_length = 19
        self.max_sen_len = 10

        self.data_set = list(pickle.load(open(os.path.join(root_dir, 'recipe1m_'+phase+'_filtered.pkl'), 'rb')))


    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        data = self.data_set[idx]

        t_inst = np.zeros((self.max_length, self.max_sen_len))
        for i in range(len(data['tokenized'])):
            input_ids = [self.tokenizer[word] for word in data['tokenized'][i]]
            end = len(input_ids)
            if end > self.max_sen_len:
                end = self.max_sen_len
            t_inst[i, :end] = input_ids[:end]
        
        return t_inst, len(data['tokenized']), data['id']


def collate_fn(data):

    t_inst, length, idx = zip(*data)

    return torch.tensor(t_inst).long(), torch.tensor(length), idx[0], torch.ones(1, 4, 10).long(), torch.ones(1)

def get_loader(root_dir, phase, batch_size, shuffle, num_workers, drop_last):
    
    data_loader = torch.utils.data.DataLoader(RecipeDataset(root_dir=root_dir, phase=phase), 
                                              batch_size=batch_size, 
                                              shuffle=shuffle, 
                                              num_workers=num_workers, 
                                              drop_last=drop_last,
                                              collate_fn=collate_fn)
    return data_loader


