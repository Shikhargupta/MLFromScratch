import nltk
import torch
import numpy as np
import pandas as pd
import json
from nltk.corpus import brown
from data import PrepareDataset
from dataset import BrownCorpus
from torch.utils.data import DataLoader
from model import Cbow
from torch import nn
import torch.optim as optim


if __name__ == '__main__':
    epochs = 10
    window = 2

    nltk.download('brown')
    data = brown.sents()

    prep_data = PrepareDataset(data=data, window=window)
    train_dataset = prep_data.get_context_target_pairs()
    train_dataset = BrownCorpus(train_dataset)


    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    params = {'batch_size': 32,
            'shuffle': True,
            'num_workers': 2}

    training_generator = DataLoader(train_dataset, **params)
    mdl = Cbow(input_size=len(prep_data.dict), hidden_size=256, window=window)

    loss_fn = nn.CrossEntropyLoss()  # binary cross entropy
    optimizer = optim.Adam(mdl.parameters(), lr=0.0001)

    for _epoch in range(epochs):
        for local_batch, local_labels in training_generator:
                # Transfer to M1 GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                y = mdl.forward(local_batch.to(torch.float32))

                loss = loss_fn(y,local_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print("Epoch : ", _epoch)
        print('Loss: ', loss)
        torch.save(mdl.state_dict(), 'model_' + str(_epoch) + '.pt')

    torch.save(mdl.state_dict(), 'model.pt')
    print("hereee")

