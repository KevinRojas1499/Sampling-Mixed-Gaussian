import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


class Score(nn.Module):

    def __init__(self,n):
        nodes = [64,64,64]
        super(Score, self).__init__()
        self.first_layer = nn.Linear(n+1, nodes[0])
        self.second_layer = nn.Linear(nodes[0], nodes[1])
        self.third_layer = nn.Linear(nodes[1], nodes[2])
        self.final_score = nn.Linear(nodes[2], n)

    def forward(self, x,t):
        x = torch.cat((x,t),dim=-1)
        x = x.float()
        
        x = F.logsigmoid(self.first_layer(x))
        x = F.logsigmoid(self.second_layer(x))
        x = F.logsigmoid(self.third_layer(x))
        x = self.final_score(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n, num_layers=2):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(n, n-1))
            else:
                self.layers.append(nn.Linear(n-1, n-1))

    def forward(self, x):
        x = x.float()
        for layer in layers:
            x = nn.ReLU(layer(x))
        return x


class Decoder(nn.Module):
    def __init__(self, n, num_layers=2):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            if i == num_layers-1:
                self.layers.append(nn.Linear(n-1, n))
            else:
                self.layers.append(nn.Linear(n-1, n-1))

    def forward(self, x):
        x = x.float()
        for layer in layers:
            x = nn.ReLU(layer(x))
        return x


class AutoEncoder(nn.Module):

    def __init__(self, n, num_layers=2):
        super().__init__()
        self.E = Encoder(n, num_layers=num_layers)
        self.D = Decoder(n, num_layers=num_layers)

    def forward(self, x):
        x_hidden = self.E(x)
        x = self.D(x_hidden)
        return x


@dataclass
class LDM:
    score: Score
    autoencoder: AutoEncoder