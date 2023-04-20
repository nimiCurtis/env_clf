import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# import cv after albumentations 
# or second solution export LD_PRELOAD=/home/zion/.local/lib/python3.8/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
import cv2

class Transformer():
    def __init__(self):
        pass

    def train_transform(self):
        train_transform = A.Compose(
            [   
                A.Crop(x_min=119,y_min=59,x_max=519,y_max=359),
                A.Resize(height=224,width=224),
                # A.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
                A.GaussNoise(mean=0.5,var_limit=[10,10],p=0.1),
                A.RandomBrightnessContrast(p=0.2),
                
                ToTensorV2(),
            ]
        )

        return train_transform

    def eval_transform(self):
        val_transform = A.Compose(
            [   
                A.Crop(x_min=119,y_min=59,x_max=519,y_max=359),
                A.Resize(height=224,width=224),
                ToTensorV2()
            ]
        )

        return val_transform

    def one_hot_transform(self,target,num_classes):
        one_hot_target = torch.zeros(num_classes)
        one_hot_target[target] = 1
        return one_hot_target
