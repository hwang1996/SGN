import pickle
from pytorch_transformers import *
from tqdm import tqdm
import torch
import numpy as np
import os

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

###############################################################################################

def select_tokens(dataset):

    dict_file_name = os.path.join('../../Ordered-Neurons/recipe_data/', 'dict_recipe.pkl')
    dictionary = Dictionary()

    train_tokenized = {}
    # for data in tqdm(dataset):
    #     for token in data['tokenized'][-1]:
    #         if token in train_tokenized:
    #             train_tokenized[token] += 1
    #         else:
    #             train_tokenized[token] = 1
    for i in tqdm(range(len(dataset))):
        for j in range(len(dataset[i]['tokenized'])):
            for k in range(len(dataset[i]['tokenized'][j])):
                if dataset[i]['tokenized'][j][k] in train_tokenized:
                    train_tokenized[dataset[i]['tokenized'][j][k]] += 1
                else:
                    train_tokenized[dataset[i]['tokenized'][j][k]] = 1
    sorted_tokenized = sorted(train_tokenized.items(), key=lambda item:item[1], reverse=True)
    selected_tokens = []
    for i in range(10000):
        selected_tokens.append(sorted_tokenized[i][0])
        dictionary.add_word(sorted_tokenized[i][0])
    print('Number of words:', len(dictionary))
    pickle.dump(dictionary, open(dict_file_name, 'wb'))

    return selected_tokens, dictionary, sorted_tokenized

def construct_subset(dataset, selected_tokens):
    subset = {}
    # for i, data in tqdm(enumerate(dataset)):
    #     count = 0
    #     total_count = 0
    #     for token in data['tokenized'][-1]:
    #         if token in selected_tokens:
    #             count += 1
    #         total_count += 1

    for i in tqdm(range(len(dataset))):
        count = 0
        total_count = 0
        for j in range(len(dataset[i]['tokenized'])):
            for k in range(len(dataset[i]['tokenized'][j])):
                if dataset[i]['tokenized'][j][k] in selected_tokens:
                    count += 1
                total_count += 1
        if count == total_count:
            subset[i] = dataset[i]

    return subset

###############################################################################################

if __name__ == '__main__':

    data_path = 'data/'
    if os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = pickle.load(open('data/recipe1m_train_filtered.pkl', 'rb')) 

    selected_tokens, dictionary, _ = select_tokens(dataset)
    subset = construct_subset(dataset, selected_tokens)

    sub_feature_set = {}
    pad_ = torch.zeros(1, 768)

    dict_file_name = os.path.join(data_path, 'dict_recipe.pkl')
    if os.path.exists(dict_file_name):
        dictionary = pickle.load(open(dict_file_name, 'rb'))
    else:
    dictionary = Dictionary()
    dictionary.add_word('<eos>')
    for value in subset.values():
        for word in value['tokenized'][-1]:
            # import pdb; pdb.set_trace()
            dictionary.add_word(word)
    print('Number of words:', len(dictionary))
    pickle.dump(dictionary, open(dict_file_name, 'wb'))

