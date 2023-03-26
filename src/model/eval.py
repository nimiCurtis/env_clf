import os
import sys
from tqdm import tqdm
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# import from parallel modules
PATH = os.path.join(os.path.dirname(__file__),'../')
sys.path.insert(0, PATH)
from data.dataset import EnvDataset
from model.pre_process import Transformer
from model.utils.visualize import visualize_augmentations
from model.utils.metrices import MetricMonitor
from model import models


def calculate_accuracy(output, target):
    # Get the predicted class index for each sample
    _, predicted = torch.max(output, dim=1)
    _, target = torch.max(target, dim=1)
    # Compare the predicted class with the target class and calculate the number of correct predictions
    correct = (predicted == target).sum().item()
    # Calculate the accuracy
    accuracy = correct / len(target)
    return accuracy

def evaluate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True)
            target = target.to(params["device"], non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            accuracy = calculate_accuracy(output, target)

            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            stream.set_description(
            "\033[34mEpoch: {epoch}.\033[0m \033[32mTest.      {metric_monitor}\033[0m".format(epoch=epoch, metric_monitor=metric_monitor)
        )
