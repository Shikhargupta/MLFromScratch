import numpy as np
import string
import re

class PrepareDataset:
    def __init__(self, data, window=2):
        self.data = data
        self.window = window
        self.dict = None
        self.dataset = None
        self.size = None
        self.reverse_dict = None

    def prepare_dict(self):
        if self.dict is not None: return self.dict
        flat_map = [word.lower() for sentence in self.data for word in sentence]
        _set = set(flat_map)
        self.dict = {val:idx for idx, val in enumerate(_set)}
        self.reverse_dict = {idx:val for idx, val in enumerate(_set)}
        self.size = len(_set)
        return self.dict

    def clean_data(self):
        tmp = []
        for sentence in self.data:
            new_sentence = ' '.join(word for word in sentence)
            new_sentence = new_sentence.translate(str.maketrans('', '', string.punctuation))
            new_sentence = re.sub(' +', ' ', new_sentence.strip())
            tmp.append(new_sentence.split(' '))
        self.data = tmp
        del tmp
        return

    def get_context_target_pairs(self):
        if self.dataset is not None: return self.dataset
        self.clean_data()
        self.prepare_dict()
        self.dataset = []
        for sentence in self.data:
            length = len(sentence)
            for idx, target in enumerate(sentence):
                context = list(range(max(0,idx-self.window),idx))
                context.extend(list(range(min(length-1, idx+1), min(length-1, idx+self.window+1))))
                if idx in context: context.remove(idx)

                context_words = []
                for _id in context:
                    context_words.append(sentence[_id].lower())
                self.dataset.append({'context':context_words, 'target':target.lower()})
        return self.dataset


