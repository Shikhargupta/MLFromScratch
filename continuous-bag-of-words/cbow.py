import nltk
import torch
import numpy as np
import pandas as pd
from nltk.corpus import brown
from data import PrepareDataset

nltk.download('brown')
data = brown.sents()

prep_data = PrepareDataset(data=data, window=2)
prep_data.prepare_data()
print("hereee")