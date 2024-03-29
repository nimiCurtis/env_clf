import os
import sys
from tqdm import tqdm
import warnings
import numpy as np
import torch
from torch import nn, optim, manual_seed, save, cuda, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, WeightedRandomSampler

# import from parallel modules
PATH = os.path.join(os.path.dirname(__file__),'../')
sys.path.insert(0, PATH)
from data.dataset import EnvDataset
from model.pre_process import Transformer
from model.utils.visualize import visualize_augmentations, show_batch
from model.utils.metrices import MetricMonitor
from model.utils.early_stopping import EarlyStopper
from model.utils.env_label import EnvLabel
from model.eval import calculate_accuracy, evaluate
from model import models

import hydra
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("path", lambda : PATH)
import wandb 
    
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
        cuda.empty_cache()
        output = model(images)
        # Calculate the loss between the model's output and the target labels
        
        if params.debug and i%10==0:
                show_batch(images=images,labels=target,predictions=output,step=i,evaluating_loop=True)
        
        
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
        
    # Return the average loss and accuracy computed by the MetricMonitor object
    return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Accuracy']['avg']

@hydra.main( version_base=None ,config_path="../../config/env_clf_config", config_name = "env_clf")
def main(cfg:DictConfig):
    
    if cfg.wandb.run.enable:
        run = wandb.init(entity=cfg.wandb.setup.entity, project=cfg.wandb.setup.project, name=cfg.wandb.run.name)
    
    # Set the hyperparameters for the experiment based on the config
    dataset_conf = cfg.dataset
    model_conf = cfg.model
    training_conf = cfg.training
    optimizer_conf = cfg.optimizer
    criterion_conf = cfg.criterion

    # Set models directory for saving the model
    models_dir = os.path.join(PATH,'../models')
    if not(os.path.exists(os.path.join(models_dir,model_conf.name))):
        os.mkdir(os.path.join(models_dir,model_conf.name))

    # Set manual seed for reproducibility
    manual_seed(dataset_conf.seed)
    
    # Define the data transformations using the custom Transformer class
    transformer = Transformer()
    
    # Load the training and testing datasets
    train_dataset = EnvDataset(root=PATH+dataset_conf.train_dataset_path,
                               dim1=model_conf.dim1,
                               idx_exclude=dataset_conf.idx_exclude,
                                fraction=dataset_conf.subset_fraction,
                                transform=transformer.train_transform(),
                                target_transform=transformer.one_hot_transform)
    
    if dataset_conf.subset_fraction != 1.0:
        train_dataset = train_dataset.get_subset_dataset().dataset


    test_dataset = EnvDataset(root=PATH+dataset_conf.test_dataset_path,
                                idx_exclude=dataset_conf.idx_exclude,
                                dim1=model_conf.dim1,
                                transform=transformer.eval_transform(),
                                target_transform=transformer.one_hot_transform)


    #set random sampler
    sampler_weights = train_dataset.class_sampler_weights
    (sampler, shuffle) = (WeightedRandomSampler(weights=sampler_weights,num_samples=len(train_dataset)), False) if training_conf.balance_sampler else (None, True)
    
    # Create data loaders to load the datasets in batches
    train_loader = DataLoader(train_dataset , batch_size=dataset_conf.train_batch_size,sampler=sampler, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, dataset_conf.test_batch_size, shuffle=False)
    
    # Initialize min loss
    min_loss = float('inf')

    # Initialize train/test total loss lists
    epoch_train_total_loss = []
    epoch_test_total_loss = []
    
    # Initialize train/test total accuracies lists
    epoch_train_total_acc = []
    epoch_test_total_acc = []
    
        # Load the specified model architecture
    model = getattr(models, model_conf.name)(version=model_conf.version,
                                            dim1=model_conf.dim1,
                                            num_classes=train_dataset.num_classes,
                                            classifier_cfg=model_conf.classifier_layer )
    
    # Move the model to the specified device (CPU or GPU)
    model = model.to(training_conf.device)

    # Set balance/imbalnce data
    # Define the loss function based on the criterion name
    class_weights = train_dataset.class_loss_weights if criterion_conf.weight_loss else None
    criterion = getattr(nn, criterion_conf.name)(weight = class_weights).to(training_conf.device)

    # Get the optimizer object based on the optimizer name
    optimizer = getattr(optim, optimizer_conf.name)(model.parameters(), lr=optimizer_conf.learning_rate)
    
    # Define early-stopper criterion
    early_stopper = EarlyStopper(patience=3,min_delta=0.1)
    
    # Train and evaluate the model for the specified number of epochs
    for epoch in range(1, training_conf.num_epochs + 1):
        epoch_train_loss, epoch_train_acc = train(train_loader, model, criterion, optimizer, epoch, training_conf) # train the model on the training set
        epoch_test_loss, epoch_test_acc = evaluate(test_loader, model, criterion, epoch, training_conf) # evaluate the model on the testing set
        
        if cfg.wandb.run.enable:
            wandb.log({"train_loss": epoch_train_loss,
                    "train_accuray": epoch_train_acc,
                    "test_loss": epoch_test_loss,
                    "test_accuracy": epoch_test_acc})


        # Add losses to the relevant list
        epoch_train_total_loss.append(epoch_train_loss)
        epoch_test_total_loss.append(epoch_test_loss)

        # Add accuracies to the relevant list
        epoch_train_total_acc.append(epoch_train_acc)
        epoch_test_total_acc.append(epoch_test_acc)
        
        # Save model
        if training_conf.save and epoch_test_loss < min_loss:
            save(model.state_dict(),os.path.join(models_dir,training_conf.save_path+'.pt'))

        if early_stopper.early_stop(validation_loss=epoch_test_loss):
            print("[INFO]  Early stopping..")
            break


if __name__=='__main__':
    main()
