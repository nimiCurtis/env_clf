"""
TODO:
- make generic with data type. not only depth.
- change the save img depend on the data type.
"""

import os
import sys
from omegaconf import OmegaConf

import numpy as np
import cv2
import pandas as pd

from torch.utils.data import Dataset

PATH = os.path.join(os.path.dirname(__file__),'../../../')
sys.path.insert(0, PATH)
from bag_utils.modules.bag_parser.bag_parser import Parser
from bag_utils.modules.bag_reader.bag_reader import BagReader
from bag_utils.modules.utils.image_data_handler import DepthHandler
dp = DepthHandler()

from env_clf.src.utils.env_label import EnvLabel


DATASET = os.path.join(os.path.dirname(__file__),'../../dataset/real')

def split_to_train_test(bag_batch_folder):
    config_path = os.path.join(bag_batch_folder,'saved_configs/recorder_configs/.hydra/config.yaml')
    cfg = OmegaConf.load(config_path)

    set = cfg.recording.set
    label_num = cfg.recording.label
    label_name = EnvLabel(label_num).name

    bag_obj = BagReader()

    for filename in os.scandir(bag_batch_folder): 
            if filename.is_file() and filename.path.split('.')[-1]=='bag':
                bag_file = filename.path
                bag_obj.bag = bag_file
                move_to(bag_obj, set=set, label = label_name)


def move_to(bag_obj:BagReader,set,label):
    
    try:
    
        if os.path.exists(bag_obj.MetaData["depth"]):
            df = pd.read_csv(bag_obj.MetaData["depth"])

        output_folder = os.path.join(os.path.join(DATASET,set),label)
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
            output_filename = os.path.join(output_folder, f"{label}_{num_imgs+index}.jpg")

            # Save the image as a .jpg file using OpenCV
            cv2.imwrite(output_filename, cv_img)


class EnvDataset(Dataset):
    
    def __init__(self):
        pass
    
    def __len__(self):
        pass

    def __getitem__(self,idx):
        pass


def main():
    # get arguments
    parser = Parser.get_parser()
    args = Parser.get_args(parser)
    try:
    
        split_to_train_test(args.bag_batch_folder)
    
    except TypeError as e:
        print(e)
        print("Don't forget to insert bag_batch_folder argument!")


if __name__=='__main__':
    main()
