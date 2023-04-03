import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import defaultdict


# define the EnDNET model
class EnDNet(nn.Module):
    def __init__(self):
        super(EnDNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16*112*112, out_features=3)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16*112*112)
        x = self.fc1(x)
        return x

class ResNet(nn.Module):
    def __init__(self, version='resnet18', pretrained=False, num_classes=3) -> None:
        super(ResNet, self).__init__()
        self.version = version
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.resnet = self.get_resnet()
        self.fc = nn.Linear(self.resnet.fc.out_features, self.num_classes)
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        x = F.relu(x)
        x = F.softmax(x)
        return x
    
    def get_resnet(self):
        if self.version == 'resnet18':
            resnet = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'resnet34':
            resnet = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'resnet50':
            resnet = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'resnet101':
            resnet = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'resnet152':
            resnet = getattr(models, self.version)(pretrained=self.pretrained)
        else:
            raise ValueError('Invalid ResNet version specified')
        return resnet

