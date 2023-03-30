import os
import sys
from tqdm import tqdm
from torch import nn, optim, manual_seed
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

import hydra
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_resolver("path", lambda : PATH)

def train(train_loader, model:nn.Module, criterion, optimizer:Optimizer, epoch, params):
    # Create a MetricMonitor object to keep track of the loss and accuracy
    metric_monitor = MetricMonitor()
    # Set the model to training mode
    model.train()
    # Create a progress bar using the tqdm library
    stream = tqdm(train_loader)
    # Loop over each batch of data in the training dataset
    for i, (images, target) in enumerate(stream, start=1):
        # Move the input data and target labels to the GPU (if available)
        images = images.to(params.device, non_blocking=True)
        target = target.to(params.device, non_blocking=True)
        # Forward pass through the model to get the output predictions
        output = model(images)
        # Calculate the loss between the model's output and the target labels
        loss = criterion(output, target)
        # Calculate the accuracy of the model's predictions
        accuracy = calculate_accuracy(output, target)
        # Update the MetricMonitor object with the current loss and accuracy
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)
        # Zero out the gradients in the optimizer
        optimizer.zero_grad()
        # Compute the gradients of the loss with respect to the model's parameters
        loss.backward()
        # Use the optimizer to update the model's parameters based on the computed gradients
        optimizer.step()
        # Update the progress bar with the current epoch, batch, and metrics
        stream.set_description(
            "\033[34mEpoch: {epoch}.\033[0m \033[36mTrain.      {metric_monitor}\033[0m".format(epoch=epoch, metric_monitor=metric_monitor)
        )
        
@hydra.main( version_base=None ,config_path="../../config/env_clf_config", config_name = "env_clf")
def main(cfg:DictConfig):
    
    # Set the hyperparameters for the experiment based on the config
    dataset_conf = cfg.dataset
    model_conf = cfg.model
    training_conf = cfg.training
    optimizer_conf = cfg.optimizer
    criterion_conf = cfg.criterion
    
    # Set manual seed for reproducibility
    manual_seed(dataset_conf.seed)
    
    # Load the specified model architecture
    model = getattr(models, model_conf.name)()
    
    # Move the model to the specified device (CPU or GPU)
    model = model.to(training_conf.device)

    # Define the loss function based on the criterion name
    criterion = getattr(nn, criterion_conf.name)().to(training_conf.device)
    
    # Get the optimizer object based on the optimizer name
    optimizer = getattr(optim, optimizer_conf.name)(model.parameters(), lr=optimizer_conf.learning_rate)
    
    # Define the data transformations using the custom Transformer class
    transformer = Transformer()
    
    # Load the training and testing datasets
    train_dataset = EnvDataset(root=PATH+'../dataset/real/train',
                                transform=transformer.train_transform(),
                                target_transform=transformer.one_hot_transform)

    test_dataset = EnvDataset(root=PATH+'../dataset/real/test',
                                transform=transformer.eval_transform(),
                                target_transform=transformer.one_hot_transform)
    
    # Create data loaders to load the datasets in batches
    train_loader = DataLoader(train_dataset, batch_size=dataset_conf.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, dataset_conf.test_batch_size, shuffle=False)
    
    # Train and evaluate the model for the specified number of epochs
    for epoch in range(1, training_conf.num_epochs + 1):
        train(train_loader, model, criterion, optimizer, epoch, training_conf) # train the model on the training set
        evaluate(test_loader, model, criterion, epoch, training_conf,debug=True) # evaluate the model on the testing set

if __name__=='__main__':
    main()