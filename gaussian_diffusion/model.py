import torch
import torch.nn as nn
import torch.nn.functional as F


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