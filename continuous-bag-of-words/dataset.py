from torch.utils.data import Dataset
import json
import numpy as np
import torch

class BrownCorpus(Dataset):
    def __init__(self, data, model):
        self.data = data
        self.model=model
        with open('/Users/shikhar/Desktop/Personal/Misc/MLFromScratch/continuous-bag-of-words/words_dictionary.json') as f:
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
        while len(x) < 4:
            x.append([0]*self.size)
        Y = self.get_one_hot(self.data[idx]['target'])
        return torch.from_numpy(np.array(x)),torch.from_numpy(np.array(Y))
