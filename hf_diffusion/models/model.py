import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleScore(nn.Module):
    def __init__(self, n_channels=3):
        super().__init__()
        self.n_channels = n_channels
        # Input channels, output channels, kernel size. Assumes 8x8 grid
        self.conv1 = nn.Conv2d(self.n_channels, 8, 3)
        # kernel size, stride
        self.pool = nn.MaxPool2d(2, 1)
        self.conv2 = nn.Conv2d(8, 16, 3)
        # Plug in time into start of linear head
        self.fc1 = nn.Linear(16 * 2 * 2 + 1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)

    def forward(self, x, t):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.cat((x, t.view((-1, 1))), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, 1, 8, 8)