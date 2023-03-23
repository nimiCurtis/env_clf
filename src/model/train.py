import os
import sys

import torchvision
from torch.utils.data import Dataset, DataLoader

# import from parallel modules
PATH = os.path.join(os.path.dirname(__file__),'../')
sys.path.insert(0, PATH)
from data.dataset import EnvDataset


# train_dataset = torchvision.datasets.ImageFolder(root='/home/roblab20/catkin_ws/src/Exo_Intent/env_clf/dataset/real/train')
# test_dataset = torchvision.datasets.ImageFolder(root='/home/roblab20/catkin_ws/src/Exo_Intent/env_clf/dataset/real/test')
train_dataset = EnvDataset(root='/home/roblab20/catkin_ws/src/Exo_Intent/env_clf/dataset/real/train',target_transform=EnvDataset.one_hot_transform)


train_loader = DataLoader(train_dataset)
# test_loader = DataLoader(test_dataset)

a=1