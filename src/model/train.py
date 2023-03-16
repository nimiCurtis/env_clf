import torchvision
from torch.utils.data import Dataset, DataLoader

train_dataset = torchvision.datasets.ImageFolder(root='/home/roblab20/catkin_ws/src/Exo_Intent/env_clf/dataset/real/train')
valid_dataset = torchvision.datasets.ImageFolder(root='/home/roblab20/catkin_ws/src/Exo_Intent/env_clf/dataset/real/test')

train_loader = DataLoader(train_dataset)
valid_loader = DataLoader(valid_dataset)

a=1