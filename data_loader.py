from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
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

    def __init__(self, root_dir='../../inversecooking/data/recipe1m', phase='train'):

        dict_file_name = os.path.join('../../Ordered-Neurons/recipe_data/', 'dict_recipe.pkl')
        self.tokenizer = pickle.load(open(dict_file_name, 'rb'))
        # self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.max_length = 19
        self.max_sen_len = 10
        self.select_num = 3

        if phase == 'train':
            self.data_set = list(pickle.load(open(os.path.join(root_dir, 'recipe1m_train_5.pkl'), 'rb')))
        else:
            self.data_set = list(pickle.load(open(os.path.join(root_dir, 'recipe1m_test_5.pkl'), 'rb')))[:5000]

        # all_fea_set = list(pickle.load(open(os.path.join(root_dir, 'sub_13_15.pkl'), 'rb')).values())
        
        # sample_num = len(all_fea_set)
        # train_num = int(0.8*sample_num)
        # if phase == 'train':
        #     self.fea_set = all_fea_set[:train_num]
        # else:
        #     self.fea_set = all_fea_set[train_num:]

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):

        data = self.data_set[idx]

        max_choice = len(data['tokenized'])-1 if len(data['tokenized'])-1<self.max_length else self.max_length-1
        sen_pos = random.randint(4, max_choice)
        # sen_pos = 15
        selections = random.sample(np.arange(sen_pos).tolist(), self.select_num)

        t_inst = np.zeros((self.max_length, self.max_sen_len))
        for i in range(sen_pos):
            # if i >= self.max_length:
            #     break
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
        # target_ids = np.array(target_ids[:self.max_sen_len])

        target = random.randint(0, 3)
        cand_ids = np.insert(random_sen, target, target_sen, 0)
        # import pdb; pdb.set_trace()

        return torch.tensor(t_inst).long(), torch.tensor(cand_ids).long(), target

def collate_fn(data):

    t_inst, cand_ids, target = zip(*data)
    cand_ids = torch.stack(cand_ids)
    input_ids = torch.stack(t_inst)

    # lengths = [len(cap) for cap in t_inst]
    # input_ids = torch.zeros(len(t_inst), max(lengths), t_inst[0].size(1)).long()
    # for i, cap in enumerate(t_inst):
    #     end = lengths[i]
    #     input_ids[i, :end, :] = cap[:end, :] 
    # import pdb; pdb.set_trace()

    return input_ids, cand_ids.long(), torch.tensor(target).long()

def get_loader(phase, batch_size, shuffle, num_workers, drop_last):
    
    data_loader = torch.utils.data.DataLoader(RecipeDataset(phase=phase), 
                                              batch_size=batch_size, 
                                              shuffle=shuffle, 
                                              num_workers=num_workers, 
                                              drop_last=drop_last,
                                              collate_fn=collate_fn)
    return data_loader