'''
TODO: 
- defualt model - takes the first model on a specific dir
- num_classes - based on cfg
'''

import os
import sys
import warnings
import argparse
from omegaconf import OmegaConf

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
from model.utils.visualize import show_image
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

def main():
    
    parser = argparse.ArgumentParser(description ='Environment context recognition')
    parser.add_argument('--dataset', '-d', default=PATH+'../dataset/real/test',
                    help='dataset folder (default: test set folder)')
    
    parser.add_argument('--model', '-m', default=PATH+'../models/ResNet/v_resnet34_2023-04-17_15-20-22.pt',
                    help='model architecture (default: resnet18)')

    parser.add_argument('--image','-i',dest='image_number',default=0, type=int, help='Image number')
    parser.add_argument('--gpu',action='store_true', help='Use GPU if available (default: 0)')


    # Parse the arguments
    args = parser.parse_args()

    # Define the data transformations using the custom Transformer class
    transformer = Transformer()

    # Load the model
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
    
    # decode the parent model
    parent_model = args.model.rsplit('/')[-2]
    # decode the version 
    model_version = args.model.rsplit('/')[-1].rsplit('.')[0]
    # take the cfg
    model_cfg_path = os.path.join(os.path.join(PATH+'../config/saved_configs/env_clf_config',model_version),'.hydra/config.yaml')
    model_cfg = OmegaConf.load(model_cfg_path)
    model = getattr(models, parent_model)(version=model_cfg.model.version,num_classes=model_cfg.training.num_classes)
    model.load_state_dict(torch.load(args.model))
    model.to(device=device)
    model.eval()

    # Load the inference dataset
    inference_dataset = InferenceDataset(root=args.dataset,
                                transform=transformer.train_transform())
    # load the specific image
    img_tensor = inference_dataset[args.image_number]

    # unsqueeze the tensor iamge for model adaptability
    img_tensor = img_tensor.unsqueeze(0)

    
    with torch.no_grad():
        output = model(img_tensor)
    
    predicted_label = torch.argmax(output)
    prob = torch.max(output)
    predicted_label = predicted_label.detach().numpy()
    show_image(image = img_tensor[0],image_number=args.image_number,pred_label= predicted_label,prob=prob.detach().float())

if __name__ == '__main__':
    main()