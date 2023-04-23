import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import defaultdict
torch.cuda.empty_cache()

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
        x = F.softmax(x,dim=1)
        return x

class VGG(nn.Module):
    def __init__(self, version='vgg16', pretrained=False, num_classes=3) -> None:
        super(VGG, self).__init__()
        self.version = version
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.vgg = self.get_vgg()
        self.backbone = self.vgg.features
        
        self.layer1 = nn.Linear(in_features=512*7*7,out_features=4000)
        self.layer2 = nn.Linear(in_features=1000,out_features=self.num_classes)

        
        
    def forward(self, x):
        x = self.vgg(x)
        # x = x.view(-1,512*7*7)
        x = self.layer2(x)
        # x = F.relu(x)
        # x=self.layer2(x)
        x = F.relu(x)
        x = F.softmax(x,dim=1)
        return x
    
    def get_vgg(self):
        if self.version == 'vgg11':
            vgg = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'vgg11_bn':
            vgg = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'vgg13':
            vgg = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'vgg13_bn':
            vgg = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'vgg16':
            vgg = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'vgg16_bn':
            vgg = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'vgg19':
            vgg = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'vgg19_bn':
            vgg = getattr(models, self.version)(pretrained=self.pretrained)
        else:
            raise ValueError('Invalid VGG version specified')
        return vgg

class ResNet(nn.Module):
    def __init__(self, version='resnet18', pretrained=False, num_classes=3) -> None:
        super(ResNet, self).__init__()
        self.version = version
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.resnet = self.get_resnet()
        self.backbone = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        # self.backbone = self.resnet
        self.layer1 = nn.Linear(in_features= 512, out_features=1000,bias=True)
        self.layer2 = nn.Linear(in_features=1000 ,out_features=self.num_classes)

        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, 512*1*1)
        x = self.layer1(x)
        x=self.layer2(x)
        x = F.relu(x)
        x = F.softmax(x,dim=1)
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


class ResNext(nn.Module):
    def __init__(self, version='resnext50_32x4d', pretrained=False, num_classes=3) -> None:
        super(ResNext, self).__init__()
        self.version = version
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.resnext = self.get_resnext()
        self.backbone = torch.nn.Sequential(*(list(self.resnext.children())[:-1]))
        # self.backbone = self.resnet
        self.layer1 = nn.Linear(in_features= 2048, out_features=1000,bias=True)
        self.layer2 = nn.Linear(in_features=1000 ,out_features=self.num_classes)

        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, 2048*1*1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = F.softmax(x,dim=1)
        return x
    
    def get_resnext(self):
        if self.version == 'resnext50_32x4d':
            resnext = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'resnext101_64x4d':
            resnext = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'resnext101_32x8d':
            resnext = getattr(models, self.version)(pretrained=self.pretrained)
        else:
            raise ValueError('Invalid ResNext version specified')
        return resnext



class EfficientNet(nn.Module):
    def __init__(self, version='efficientnet_b0', pretrained=False, num_classes=3) -> None:
        super(EfficientNet, self).__init__()
        self.version = version
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.efficientnet = self.get_efficientnet()
        self.backbone = torch.nn.Sequential(*(list(self.efficientnet.children())[:-1]))
        # self.backbone = self.resnet
        self.layer1 = nn.Linear(in_features= 1280, out_features=500,bias=True)
        self.layer2 = nn.Linear(in_features=500 ,out_features=self.num_classes)
        self.backbone
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, 1280*1*1)
        x = self.layer1(x)
        x=self.layer2(x)
        x = F.relu(x)
        x = F.softmax(x,dim=1)
        return x
    
    def get_efficientnet(self):
        if self.version == 'efficientnet_b0':
            efficientnet = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'efficientnet_b4':
            efficientnet = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'efficientnet_b5':
            efficientnet = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'efficientnet_b6':
            efficientnet = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'efficientnet_b7':
            efficientnet = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'efficientnet_v2_s':
            efficientnet = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'efficientnet_v2_s':
            efficientnet = getattr(models, self.version)(pretrained=self.pretrained)
        elif self.version == 'efficientnet_v2_m':
            efficientnet = getattr(models, self.version)(pretrained=self.pretrained)
        else:
            raise ValueError('Invalid EfficientNet version specified')
        return efficientnet

