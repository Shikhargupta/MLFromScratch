import string

class BagofWords:
    def __init__(self, sentences=None, vocab=None):
        self.sentences = [s.lower() for s in sentences]
        if vocab is None:
            self.vocab = self.prepare_vocab()
        else:
            self.vocab = vocab
        self.count = len(self.vocab)
    
    def prepare_vocab(self):
        '''
        Prepares vocabulary for the given set of sentences 

        Inputs:
            None
        Outputs:
            vocab : dictionary with words as key and their corresponding
                    position as value
        '''
        v = {}
        _setA = set([])
        for s in self.sentences:
            s = s.translate(str.maketrans('', '', string.punctuation))
            _setB = set(s.split(' '))
            _setA = _setA.union(_setB)
        
        idx=0
        for it in _setA:
            v[it] = idx
            idx += 1

        self.count = len(v)
        return v


    def get_all_vectors(self):
        '''
        Gets BoW representation of all the sentences

        Inputs:
            None
        
        Outputs:
            res : list of vectors representing the sentences
        '''
        res = []
        for s in self.sentences:
            s = s.translate(str.maketrans('', '', string.punctuation))
            tmp = [0]*self.count
            for word in s.split(' '):
                try:
                    tmp[self.vocab[word]] += 1
                except:
                    raise Exception("Cannot form a vector")
            res.append(tmp)

        return res

    def add_sentences(self, l):
        '''
        Adds new sentences to the database

        Inputs:
            l    : list of new sentences
        
        Outputs:
            None
        '''
        l = [_l.lower() for _l in l]
        self.sentences.extend(l)
        self.vocab = self.prepare_vocab()
        self.count = len(self.vocab)

        return
                
    def get_bow_vectors(self, s):
        '''
        Gets vector representation of provided sentences

        Inputs:
            s    : list of sentences
        
        Outputs:
            res  : list of vector representations 
        '''
        res = []
        for _s in s:
            tmp = list(self.count, 0)
            for word in _s.split(' '):
                try:
                    tmp[self.vocab[word]] += 1
                except:
                    raise Exception("Cannot form a vector")
            res.append(tmp)

        return res

