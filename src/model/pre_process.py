import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

class Transformer():
    def __init__(self):
        pass

    def train_transform(self):
        train_transform = A.Compose(
            [
                A.RandomBrightnessContrast(p=0.3),
                ToTensorV2(),
            ]
        )

        return train_transform

    def eval_transform(self):
        val_transform = A.Compose(
            [
                A.RandomBrightnessContrast(p=0.3),
                ToTensorV2()
            ]
        )

        return val_transform

    def one_hot_transform(self,target,num_classes):
        one_hot_target = torch.zeros(num_classes)
        one_hot_target[target] = 1
        return one_hot_target
