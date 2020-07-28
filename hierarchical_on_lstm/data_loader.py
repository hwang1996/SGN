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
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
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
        self.select_num = 3

        self.data_set = list(pickle.load(open(os.path.join(root_dir, 'recipe1m_'+phase+'_filtered.pkl'), 'rb')))

        ## use data with over 5 sentences for training
        self.data_set_5 = []
        for i in range(len(self.data_set)):
            if len(self.data_set[i]['tokenized']) >= 5:
                self.data_set_5.append(self.data_set[i])

        ## for quick evaluation
        if phase != 'train':
            self.data_set_5 = self.data_set_5[:5000]


    def __len__(self):
        return len(self.data_set_5)

    def __getitem__(self, idx):

        data = self.data_set_5[idx]

        max_choice = len(data['tokenized'])-1 if len(data['tokenized'])-1<self.max_length else self.max_length-1
        sen_pos = random.randint(4, max_choice)
        selections = random.sample(np.arange(sen_pos).tolist(), self.select_num)

        t_inst = np.zeros((self.max_length, self.max_sen_len))
        for i in range(sen_pos):
            input_ids = [self.tokenizer[word] for word in data['tokenized'][i]]
            end = len(input_ids)
            if end > self.max_sen_len:
                end = self.max_sen_len
            t_inst[i, :end] = input_ids[:end]
        random_sen = t_inst[selections]
        target_ids = [self.tokenizer[word] for word in data['tokenized'][sen_pos]]
        target_sen = np.zeros((1, self.max_sen_len))
        end = self.max_sen_len if len(target_ids)>self.max_sen_len else len(target_ids)
        target_sen[0, :end] = target_ids[:end]

        target = random.randint(0, 3)
        cand_ids = np.insert(random_sen, target, target_sen, 0)

        return torch.tensor(t_inst).long(), torch.tensor(cand_ids).long(), target

def collate_fn(data):

    t_inst, cand_ids, target = zip(*data)
    cand_ids = torch.stack(cand_ids)
    input_ids = torch.stack(t_inst)

    return input_ids, cand_ids.long(), torch.tensor(target).long()

def get_loader(root_dir, phase, batch_size, shuffle, num_workers, drop_last):
    
    data_loader = torch.utils.data.DataLoader(RecipeDataset(root_dir=root_dir, phase=phase), 
                                              batch_size=batch_size, 
                                              shuffle=shuffle, 
                                              num_workers=num_workers, 
                                              drop_last=drop_last,
                                              collate_fn=collate_fn)
    return data_loader