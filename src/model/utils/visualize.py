import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import copy
import numpy as np


def show_image(image,image_number,pred_label,prob=None):
    # convert to numpy image
    img = image.detach().numpy()
    # transpose dims
    img = np.transpose(img, (1, 2, 0))
    # convert to int
    img = (img).astype(np.uint8)

    fig, ax = plt.subplots()
    fig.suptitle(f"Predicted class: {pred_label} (p={prob:.2f})| Image number: {image_number}", fontsize=10)
    ax.imshow(img)
    plt.show()


def show_batch(images, labels, predictions,step,cols=8,evaluating_loop=False):
    # Get batch size
    batch_size = len(images)
    rows = int(batch_size // cols)

    
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(14, 2*batch_size))
    fig.suptitle(f"Actual vs Predicted | Step: {step}", fontsize=10)

    # Move images tensor from GPU to CPU and convert to numpy array
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    # Create grid of images
    images = np.transpose(images, (0, 2, 3, 1))
    
    # Loop through each image and label, and display actual vs predicted label in title
    for i in range(batch_size):
        image = (images[i]).astype(np.uint8)
        actual_label = labels[i]
        predicted_label = predictions[i]
        target = np.argmax(actual_label)
        predicted = np.argmax(predicted_label)
        
        correct = (predicted == target)
        
        # Set title color based on correct/wrong prediction
        if correct:
            color = 'green'
        else:
            color = 'red'
        
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(predicted_label, color=color)
        ax.ravel()[i].set_axis_off()    
        ax.ravel()[i].set_title("Label: {} Pred: {}".format(target, predicted), color=color, fontsize=6,loc='center')

    plt.tight_layout()
    if evaluating_loop:
        plt.draw()
        plt.pause(3)
        plt.close(fig)
    else:
        plt.show()


def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx+i]
        image = image.permute(1, 2, 0).numpy()
        ax.ravel()[i].imshow((image).astype(np.uint8))
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


