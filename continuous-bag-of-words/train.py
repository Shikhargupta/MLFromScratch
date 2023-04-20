import nltk
import torch
import numpy as np
import pandas as pd
import json
from nltk.corpus import brown
from data import PrepareDataset
from torch.utils.data import DataLoader

nltk.download('brown')
data = brown.sents()

prep_data = PrepareDataset(data=data, window=2)
train_dataset = prep_data.get_context_target_pairs()


if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

params = {'batch_size': 1,
          'shuffle': True}

training_generator = DataLoader(train_dataset, **params)

for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

print("hereee")

