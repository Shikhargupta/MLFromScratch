import torch
from torch import nn

class Cbow(nn.Module):

    def __init__(self, input_size=None, hidden_size=None, window=2):
        super(Cbow, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size

        self.input = nn.Linear(self.input_size, self.hidden_size)
        self.activation = nn.ReLU()
        self.hidden = nn.Linear(self.hidden_size, self.output_size)
        self.output = nn.Softmax()

    def forward(self,x):
        x = self.input(x)
        x = torch.mean(x, dim=0)

        x = self.activation(x)
        x = self.hidden(x)
        x = self.output(x)

        return x
