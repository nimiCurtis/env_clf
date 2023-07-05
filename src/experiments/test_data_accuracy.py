import sys
import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import confusion_matrix
# Import necessary modules
PATH = os.path.join(os.path.dirname(__file__), '../../../')
sys.path.insert(0, PATH)
from bag_utils.modules.bag_reader.bag_reader import BagReader
from bag_utils.modules.bag_parser.bag_parser import Parser
from bag_utils.modules.utils.image_data_handler import DepthHandler, ImageHandler
dp = DepthHandler()  # Create an instance of DepthHandler
ih = ImageHandler()  # Create an instance of ImageHandler

import hydra
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("path", lambda : PATH)

# A logger for this file
log = logging.getLogger(__name__)

def test_bag(bag_obj: BagReader, cfg:DictConfig) -> float:
    """
    Test a bag for accuracy.

    Args:
        bag_obj (BagReader): The BagReader object containing bag data.

    Returns:
        float: The accuracy of the bag.
    """

    clf_df = pd.read_csv(bag_obj.MetaData["env_clf"])  # Read environment classifier results from CSV
    pred_labels = clf_df["env_clf_res.id"].to_numpy()  # Extract predicted labels
    
    depth_synced_df = pd.read_csv(bag_obj.MetaData["depth_synced"])  # Read depth synchronized data from CSV
    gt_labels = depth_synced_df["id"].to_numpy()  # Extract ground truth labels
    data_len = len(gt_labels)
    
    
    correct = (gt_labels == pred_labels).sum()  # Count the number of correct predictions
    accuracy = (correct / data_len) * 100  # Calculate accuracy in percentage

    log.info(f"[INFO] Accuracy of bag: {bag_obj._bag_file} is:\n {round(accuracy, 3)}% ({correct}/{data_len})")


    return accuracy, correct, data_len, pred_labels, gt_labels


def test_batch(bag_obj: BagReader, batch_path: str, cfg:DictConfig) -> None:
    """
    Test a batch of bags for accuracy.

    Args:
        bag_obj (BagReader): The BagReader object containing bag data.
        batch_path (str): The path to the batch folder.

    Returns:
        None
    """
    print(f"[INFO] Testing batch: {batch_path}")
    info_dic = {"accuracy":[],
                "bags_count":0,
                "correct": 0,
                "data_len": 0,
                "gt_labels": [],
                "pred_labels": []}
    

    for filename in os.scandir(batch_path):
        if filename.is_file() and filename.path.split('.')[-1] == 'bag':
            bag_file = filename.path
            bag_obj.bag = bag_file
            accuracy, correct, data_len, gt_labels, pred_labels = test_bag(bag_obj=bag_obj, cfg=cfg) 
            info_dic["accuracy"].append(accuracy)
            info_dic["gt_labels"]+=gt_labels.tolist()
            info_dic["pred_labels"]+=pred_labels.tolist()
            info_dic["correct"]+=correct
            info_dic["data_len"]+=data_len

    accuracy_mean = np.mean(info_dic["accuracy"])
    log.info(f"Mean accuracy of batch: {batch_path} is:\n {round(accuracy_mean, 3)}% ({info_dic['correct']}/{info_dic['data_len']})")

    return  info_dic

def test_list(bag_obj: BagReader, list_batch_path: str, cfg:DictConfig) -> None:
    """
    Test a list of batch paths for accuracy.

    Args:
        bag_obj (BagReader): The BagReader object containing bag data.
        list_batch_path (str): The list of batch paths.

    Returns:
        None
    """
    print("[INFO] Testing List")
    info_dic = {"accuracy":[],
                "correct": 0,
                "data_len": 0,
                "gt_labels": [],
                "pred_labels": []}

    for bag_batch in list_batch_path:
        in_dic = test_batch(bag_obj, bag_batch, cfg)
        info_dic["accuracy"]+=in_dic["accuracy"]
        info_dic["gt_labels"]+=in_dic["gt_labels"]
        info_dic["pred_labels"]+=(in_dic["pred_labels"])
        info_dic["correct"]+=in_dic["correct"]
        info_dic["data_len"]+=in_dic["data_len"]

    accuracy_mean = np.mean(info_dic["accuracy"])
    log.info(f"Mean accuracy of total is: {round(accuracy_mean, 3)}% ({info_dic['correct']}/{info_dic['data_len']})")
    
    
    # Create confusion matrix
    labels = np.unique(np.concatenate((info_dic["gt_labels"], info_dic["pred_labels"])))
    cm = confusion_matrix(info_dic["gt_labels"], info_dic["pred_labels"])

    # Plot confusion matrix
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels,yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth Label')
    plt.title('Confusion Matrix')
    
    plt.savefig(cfg.save_dir+"confusion_matrix.png")
    
    if cfg.debug:
        plt.show()
    
    
    
    return accuracy_mean

@hydra.main( version_base=None ,config_path="../../config/experiments_config", config_name = "experiments")
def main(cfg:DictConfig):
    
    os.chdir(os.path.dirname(__file__))  # Set the current working directory to the script's directory
    
    bag_obj = BagReader()  # Create an instance of BagReader



    
    bag_file = PATH + 'env_clf/bag/bag_batch_2023-06-21_16-28-26_ALL_test/2023-06-21-16-30-41.bag'  # Default bag file


    test_list(bag_obj, list_batch_path=cfg.experiments.data, cfg=cfg)



if __name__ == '__main__':
    main()
