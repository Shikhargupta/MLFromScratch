from torch.utils.data import Dataset
import json
import numpy as np

class BrownCorpus(Dataset):
    def __init__(self, data):
        self.data = data
        with open('words_dictionary.json') as f:
            self.dict = json.load(f)
        self.size = len(self.dict)

    def __len__(self):
        return len(self.data)

    def get_one_hot(self, word):
        ohe_vec = [0]*self.size
        ohe_vec[self.dict[word.lower()]] = 1
        return ohe_vec

    def __getitem__(self, idx):
        context_words = self.data[idx]['context']
        x = []
        for _word in context_words:
            x.append(self.get_one_hot(_word))
        Y = self.get_one_hot(self.data[idx]['target'])
        return x,Y
