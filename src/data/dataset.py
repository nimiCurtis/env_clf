"""
TODO:
- make generic with data type. not only depth. (future)
- change the save img depend on the data type. (future)
"""

import os
import sys
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from collections import Counter

import numpy as np
import cv2
import pandas as pd

import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder

# Importing some custom modules from other files
PATH = os.path.join(os.path.dirname(__file__), '../../../')
sys.path.insert(0, PATH)
from bag_utils.modules.bag_parser.bag_parser import Parser
from bag_utils.modules.bag_reader.bag_reader import BagReader
from bag_utils.modules.utils.image_data_handler import DepthHandler
dp = DepthHandler()

from env_clf.src.model.utils.env_label import EnvLabel


DATASET = os.path.join(os.path.dirname(__file__), '../../dataset/real')


def split_to_train_test(bag_batch_folder):
    """
    Splits the data in a bag batch folder into train and test sets.

    Parameters:
        bag_batch_folder (str): The path to the folder containing bag files.

    Returns:
        None
    """
    # Here we construct the path to the configuration file
    config_path = os.path.join(bag_batch_folder, 'saved_configs/recorder_configs/.hydra/config.yaml')
    # We load the configuration file using the OmegaConf library
    cfg = OmegaConf.load(config_path)

    # We extract the recording set (either 'train' or 'test') from the configuration file
    set = cfg.recording.set
    label_name = cfg.recording.label

    # We create a BagReader object, which is a utility class for reading data from bag files
    bag_obj = BagReader()

    # We iterate over all files in the folder
    for filename in os.scandir(bag_batch_folder):
        # We only consider files that have the extension '.bag'
        if filename.is_file() and filename.path.split('.')[-1] == 'bag':
            bag_file = filename.path
            bag_obj.bag = bag_file
            # We check if the bag file has already been split
            if bag_obj.MetaData["in_data"]:
                print("[INFO]  Bags data already split to dataset folder")
                continue
            else:
                print("[INFO]  Bags data splitting to dataset folder")
                # We call the 'move_to' function which splits the data and moves it to the appropriate folders
                move_to(bag_obj, set=set, label=label_name)
                # We update the metadata for the bag file to indicate that it has been split
                bag_obj.update_metadata("in_data", True)


def move_to(bag_obj: BagReader, set, label):
    """
    Moves data from bag files to the appropriate train or test folders.

    Parameters:
        bag_obj (BagReader): The BagReader object representing the bag file.
        set (str): The recording set (either 'train' or 'test').
        label (str): The label name associated with the data.

    Returns:
        None
    """
    num_imgs = 0

    try:
        if os.path.exists(bag_obj.MetaData["depth"]):
            df = pd.read_csv(bag_obj.MetaData["depth"], index_col=0)

        output_folder = os.path.join(os.path.join(DATASET, set), label)
        if os.path.exists(output_folder):
            # Get a list of all files in the directory
            file_list = os.listdir(output_folder)
            # Count the number of .jpg files in the directory
            num_imgs = len([filename for filename in file_list if filename.endswith('.jpg')])

    except FileNotFoundError as e:
        print(e)

    for index, row in df.iterrows():
        # Load the .npy matrix using NumPy
        img_matrix = np.load(row['np_path'])

        # Convert the matrix to a grayscale image using OpenCV
        cv_img = dp.get_depth_normalization(img_matrix)

        # Define the output filename
        output_filename = os.path.join(output_folder, f"{label}_{num_imgs + index}.jpg")

        # Save the image as a .jpg file using OpenCV
        cv2.imwrite(output_filename, cv_img)


def dataset_distribution(dataset_folder=DATASET):
    """
    Visualizes the distribution of classes in the dataset.

    Parameters:
        dataset_folder (str): The path to the dataset folder.

    Returns:
        None
    """
    train_dir = os.path.join(dataset_folder, "train")
    test_dir = os.path.join(dataset_folder, "test")

    # Get the list of classes from the train directory
    classes = sorted(os.listdir(train_dir))

    # Count the number of images in each class for the train set
    train_counts = []
    for cls in classes:
        cls_path = os.path.join(train_dir, cls)
        count = len(os.listdir(cls_path))
        train_counts.append(count)

    # Count the number of images in each class for the test set
    test_counts = []
    for cls in classes:
        cls_path = os.path.join(test_dir, cls)
        count = len(os.listdir(cls_path))
        test_counts.append(count)

    # Calculate the total number of images in the train and test sets
    total_train = sum(train_counts)
    total_test = sum(test_counts)

    # Visualize the train and test class distributions in pie charts
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Class Distribution")

    ax1.pie(train_counts, labels=classes, autopct='%1.1f%%')
    ax1.set_title("Train Set (Total: {})".format(total_train))

    ax2.pie(test_counts, labels=classes, autopct='%1.1f%%')
    ax2.set_title("Test Set (Total: {})".format(total_test))

    # Add counts per class to the subplots
    for ax, counts in zip((ax1, ax2), (train_counts, test_counts)):
        ax.text(-1.5, 1.2, "Counts per Class", fontsize=12, weight="bold")
        ypos = 1.0
        for cls, count in zip(classes, counts):
            ax.text(-1.5, ypos, "{}: {}".format(cls, count), fontsize=10)
            ypos -= 0.1

    plt.show()


def split_all_to_train_test(bag_folder: str) -> None:
    """
    Splits all bag batch folders in the specified folder into train and test sets.

    Parameters:
        bag_folder (str): The path to the folder containing bag batch folders.

    Returns:
        None
    """
    print(f"[INFO]  Splitting folder - {bag_folder}")
    for filename in os.scandir(bag_folder):
        if filename.is_dir() and 'bag_batch' in filename.path:
            split_to_train_test(filename.path)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Finds the class folders in a dataset.

    Parameters:
        directory (str): The path to the dataset directory.

    Returns:
        Tuple[List[str], Dict[str, int]]: A tuple containing a list of class names and a dictionary mapping class names to indices.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: EnvLabel[cls_name].value for cls_name in classes}
    return classes, class_to_idx


def get_class_loss_weights(counter: Counter):
    """
    Calculates the class loss weights based on the class distribution.

    Parameters:
        counter (Counter): A Counter object containing the class distribution.

    Returns:
        List[float]: A list of class loss weights.
    """
    return [(1 - counter[id] / sum(counter.values())) for id in counter.keys()]


def get_class_sampler_weights(counter: Counter, samples):
    """
    Calculates the class sampler weights based on the class distribution.

    Parameters:
        counter (Counter): A Counter object containing the class distribution.
        samples: The samples of the dataset.

    Returns:
        List[float]: A list of class sampler weights.
    """
    return [(1 / counter[target]) for _, target in samples if target in counter.keys()]


class EnvDataset(ImageFolder):
    """
    Custom dataset class for environmental classification.

    This class extends the torchvision.datasets.ImageFolder class and provides additional functionality for handling
    class weights and class sampling.

    Parameters:
        root (str): The path to the dataset root directory.
        fraction (float, optional): The fraction of samples to use from each class (default: 1.0).
        idx_exclude (List[int], optional): List of class indexes to exclude from the dataset.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    def __init__(self, root, fraction=1.0, idx_exclude=None, loader=None, dim1=False, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        
        self.fraction = fraction
        self.dim1 = dim1
        self.idx_exclude = idx_exclude if idx_exclude is not None else []  # Handle None case
        self.num_classes = len(self.classes) # - len(self.idx_exclude)
        counter = Counter(self.targets)

        # Exclude the specified class indexes from class weights and sampler weights calculations
        for idx in self.idx_exclude:
            del counter[idx]
            # del self.class_to_idx[EnvLabel(idx).name]
            # self.classes.remove(EnvLabel(idx).name)

        self.class_loss_weights = self.get_class_weights(counter=counter)
        self.class_sampler_weights = self.get_class_sampler_weights(counter=counter, samples=self.samples)


    def __getitem__(self, index):
        path, target = self.samples[index]

        if self.dim1:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(path)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=np.array(image))["image"]
            image = image.float()

        if self.target_transform is not None:
            target = self.target_transform(target, self.num_classes)
        else:
            target = torch.tensor(target)

        return image, target

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Parameters:
            directory (str): The path to the dataset directory.

        Returns:
            Tuple[List[str], Dict[str, int]]: A tuple containing a list of class names and a dictionary mapping class names to indices.
        """
        return find_classes(directory)

    def get_subset_dataset(self):
        """
        Creates a subset of the dataset based on the specified fraction.

        Returns:
            Subset: A subset of the dataset.
        """
        subset_indices = []
        class_counts = self.targets.count(max(self.targets)) + 1

        for class_idx in range(class_counts):
            class_indices = [i for i, target in enumerate(self.targets) if target == class_idx]
            selected_indices = class_indices[:int(len(class_indices) * self.fraction)]
            subset_indices.extend(selected_indices)
    
        return Subset(self, subset_indices)

    def get_class_weights(self, counter):
        """
        Calculates the class loss weights based on the class distribution.

        Parameters:
            counter (Counter): A Counter object containing the class distribution.

        Returns:
            torch.Tensor: A tensor containing the class loss weights.
        """
        return torch.Tensor(get_class_loss_weights(counter))

    def get_class_sampler_weights(self, counter, samples):
        """
        Calculates the class sampler weights based on the class distribution.

        Parameters:
            counter (Counter): A Counter object containing the class distribution.
            samples: The samples of the dataset.

        Returns:
            List[float]: A list of class sampler weights.
        """
        return get_class_sampler_weights(counter, samples)


def main():
    # get arguments
    parser = Parser.get_parser()
    Parser.add_bool_arg(parser, name="dist", default=False)
    args = Parser.get_args(parser)
    try:
        if args.folder_of_batches is not None:
            split_all_to_train_test(args.folder_of_batches)

        if args.bag_batch_folder is not None:
            split_to_train_test(args.bag_batch_folder)
        if args.dist:
            dataset_distribution()

    except TypeError as e:
        print(e)
        print("Don't forget to insert bag_batch_folder argument!")


if __name__ == '__main__':
    main()
