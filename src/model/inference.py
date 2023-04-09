import os
import sys
from tqdm import tqdm
import warnings
import argparse

import numpy as np
import torch
from torch import nn, optim, manual_seed, save
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# import from parallel modules
PATH = os.path.join(os.path.dirname(__file__),'../')
sys.path.insert(0, PATH)
from data.dataset import EnvDataset
from model.pre_process import Transformer
from model.utils.visualize import visualize_augmentations
from model.utils.metrices import MetricMonitor
from model.eval import calculate_accuracy, evaluate
from model import models

class InferenceDataset(EnvDataset):
    def __init__(self,root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):

        path, target = self.samples[idx]
        image = self.loader(path)
        
        if self.transform is not None:
            image = self.transform(image=np.array(image))["image"]
            image = image.float()

        return image


def inference_single():
    pass

def inference_batch():
    pass

def main():
    
    parser = argparse.ArgumentParser(description ='Environment context recognition')
    parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')



# model path
# dataset folder
# device
# single idx
# batch idx


if __name__ == '__main__':
    main()