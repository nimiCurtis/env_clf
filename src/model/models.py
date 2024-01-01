import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import defaultdict
torch.cuda.empty_cache()
from timm.models import VisionTransformer
import timm
from torchvision.models import ResNet18_Weights



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

class ViT(nn.Module):
    def __init__(self, version='vit_base_patch16_224', weights=None,dim1=False, num_classes=6, classifier_cfg=None) -> None:
        super(ViT, self).__init__()
        self.version = version
        self.weights = weights
        pretrained = False if weights == None else weights
        self.num_classes = num_classes
        self.model = timm.create_model(version, pretrained=pretrained)
        if classifier_cfg is not None:
            linear1_out, linear2_out = classifier_cfg.linear1_out, classifier_cfg.linear2_out
        else:
            linear1_out, linear2_out = 256, 128
            
        self.model.head =  nn.Sequential(
            nn.Linear(self.model.head.in_features , linear1_out),
            nn.BatchNorm1d(linear1_out),
            nn.Dropout(0.2),
            nn.Linear(linear1_out , linear2_out),
            nn.Linear(linear2_out , num_classes)
        )


    def forward(self, x):
        x = self.model(x)
        x = F.softmax(x,dim=1)
        return x
    
    # def get_vit(self):
    #     if self.version == 'vgg11':
    #         vgg = getattr(models, self.version)(weights=self.weights)
    #     elif self.version == 'vgg11_bn':
    #         vgg = getattr(models, self.version)(weights=self.weights)
    #     elif self.version == 'vgg13':
    #         vgg = getattr(models, self.version)(weights=self.weights)
    #     elif self.version == 'vgg13_bn':
    #         vgg = getattr(models, self.version)(weights=self.weights)
    #     elif self.version == 'vgg16':
    #         vgg = getattr(models, self.version)(weights=self.weights)
    #     elif self.version == 'vgg16_bn':
    #         vgg = getattr(models, self.version)(weights=self.weights)
    #     elif self.version == 'vgg19':
    #         vgg = getattr(models, self.version)(weights=self.weights)
    #     elif self.version == 'vgg19_bn':
    #         vgg = getattr(models, self.version)(weights=self.weights)
    #     else:
    #         raise ValueError('Invalid VGG version specified')
    #     return vgg


## modoify VGG
class VGG(nn.Module):
    def __init__(self, version='vgg16', weights=None, num_classes=6) -> None:
        super(VGG, self).__init__()
        self.version = version
        self.weights = weights
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
            vgg = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'vgg11_bn':
            vgg = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'vgg13':
            vgg = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'vgg13_bn':
            vgg = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'vgg16':
            vgg = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'vgg16_bn':
            vgg = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'vgg19':
            vgg = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'vgg19_bn':
            vgg = getattr(models, self.version)(weights=self.weights)
        else:
            raise ValueError('Invalid VGG version specified')
        return vgg

class ResNet(nn.Module):
    def __init__(self, version='resnet18', weights=None,dim1=False, num_classes=6, classifier_cfg=None) -> None:
        super(ResNet, self).__init__()
        self.version = version
        self.weights = weights
        if classifier_cfg is not None:
            linear1_out, linear2_out = classifier_cfg.linear1_out, classifier_cfg.linear2_out
        else:
            linear1_out, linear2_out = 256, 128
            
        self.num_classes = num_classes
        self.resnet = self.get_resnet()
        self.backbone = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        
        if dim1:
            self.backbone[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.classifier_layer = nn.Sequential(
            nn.Linear(512 , linear1_out),
            nn.BatchNorm1d(linear1_out),
            nn.Dropout(0.2),
            nn.Linear(linear1_out , linear2_out),
            nn.Linear(linear2_out , num_classes)
        )



    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, 512*1*1)
        x = self.classifier_layer(x)
        x = F.softmax(x,dim=1)
        return x
    
    def get_resnet(self):
        if self.version == 'resnet18':
            resnet = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'resnet34':
            resnet = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'resnet50':
            resnet = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'resnet101':
            resnet = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'resnet152':
            resnet = getattr(models, self.version)(weights=self.weights)
        else:
            raise ValueError('Invalid ResNet version specified')
        return resnet


class ResNext(nn.Module):
    def __init__(self, version='resnext50_32x4d', weights = None, num_classes=6, classifier_cfg = None) -> None:
        super(ResNext, self).__init__()
        self.version = version
        self.weights = weights
        self.num_classes = num_classes
        self.resnext = self.get_resnext()
        self.backbone = torch.nn.Sequential(*(list(self.resnext.children())[:-1]))

        if classifier_cfg is not None:
            linear1_out, linear2_out = classifier_cfg.linear1_out, classifier_cfg.linear2_out
        else:
            linear1_out, linear2_out = 512, 256
        
        self.classifier_layer = nn.Sequential(
            nn.Linear(2048 , linear1_out),
            nn.BatchNorm1d(linear1_out),
            nn.Dropout(0.2),
            nn.Linear(linear1_out , linear2_out),
            nn.Linear(linear2_out , num_classes)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, 2048*1*1)
        x = self.classifier_layer(x)
        x = F.softmax(x,dim=1)
        return x
    
    def get_resnext(self):
        if self.version == 'resnext50_32x4d':
            resnext = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'resnext101_64x4d':
            resnext = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'resnext101_32x8d':
            resnext = getattr(models, self.version)(weights=self.weights)
        else:
            raise ValueError('Invalid ResNext version specified')
        return resnext



class EfficientNet(nn.Module):
    def __init__(self, version='efficientnet_b0', weights = None, dim1=False, num_classes=6, classifier_cfg=None) -> None:
        super(EfficientNet, self).__init__()
        self.version = version
        self.weights = weights
        self.num_classes = num_classes
        self.efficientnet = self.get_efficientnet()
        self.backbone = torch.nn.Sequential(*(list(self.efficientnet.children())[:-1]))

        if dim1:
            self.backbone[0][0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2),padding=(1,1), bias=False)

        if classifier_cfg is not None:
            linear1_out, linear2_out = classifier_cfg.linear1_out, classifier_cfg.linear2_out
        else:
            linear1_out, linear2_out = 512, 256


        self.classifier_layer = nn.Sequential(
            nn.Linear(1280 , linear1_out),
            nn.BatchNorm1d(linear1_out),
            nn.Dropout(0.2),
            nn.Linear(linear1_out , linear2_out),
            nn.Linear(linear2_out , num_classes)
        )


    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, 1280*1*1)
        x = self.classifier_layer(x)
        x = F.softmax(x,dim=1)
        return x


    def get_efficientnet(self):
        if self.version == 'efficientnet_b0':
            efficientnet = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'efficientnet_b4':
            efficientnet = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'efficientnet_b5':
            efficientnet = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'efficientnet_b6':
            efficientnet = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'efficientnet_b7':
            efficientnet = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'efficientnet_v2_s':
            efficientnet = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'efficientnet_v2_s':
            efficientnet = getattr(models, self.version)(weights=self.weights)
        elif self.version == 'efficientnet_v2_m':
            efficientnet = getattr(models, self.version)(weights=self.weights)
        else:
            raise ValueError('Invalid EfficientNet version specified')
        return efficientnet

