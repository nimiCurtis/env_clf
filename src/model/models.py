import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


# define the EnDNET model
class EnDNet(nn.Module):
    def __init__(self):
        super(EnDNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16*320*180, out_features=2)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16*180*320)
        x = self.fc1(x)
        return x

