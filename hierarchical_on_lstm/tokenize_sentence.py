import pickle
from tqdm import tqdm
import torch
import numpy as np
import os

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


###############################################################################################

if __name__ == '__main__':

    data_path = '../data/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    dict_file_name = os.path.join(data_path, 'dict_recipe.pkl')

    dataset = pickle.load(open('../data/recipe1m_train_5.pkl', 'rb')) 

    dictionary = Dictionary()

    train_tokenized = {}
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


