import sys
import os
import pandas as pd
import numpy as np

# import from parallel modules
PATH = os.path.join(os.path.dirname(__file__),'../../../')
sys.path.insert(0, PATH)
from bag_utils.modules.bag_reader.bag_reader import BagReader
from bag_utils.modules.bag_parser.bag_parser import Parser
from bag_utils.modules.utils.image_data_handler import DepthHandler, ImageHandler
dp = DepthHandler()
ih = ImageHandler()


def test_bag(bag_obj:BagReader):
    
    clf_df = pd.read_csv(bag_obj.MetaData["env_clf"])
    pred_labels = clf_df["env_clf_res.id"].to_numpy()

    depth_synced_df = pd.read_csv(bag_obj.MetaData["depth_synced"])
    gt_labels = depth_synced_df["id"].to_numpy()
    
    correct = (gt_labels==pred_labels).sum()
    
    accuracy = (correct/len(gt_labels))*100
    
    print(f"[INFO] Accuracy of bag:{bag_obj._bag_file} is:\n {round(accuracy,3)}%")
    
    return accuracy    


def test_batch(bag_obj:BagReader,batch_path:str)->None:
    print(f"[INFO]  Testing batch: {batch_path}")
    accuracy_arr = []
    for filename in os.scandir(batch_path): 
        if filename.is_file() and filename.path.split('.')[-1]=='bag':
                    bag_file = filename.path
                    bag_obj.bag = bag_file
                    accuracy_arr.append(test_bag(bag_obj))
    accuracy_mean = np.mean(accuracy_arr)
    print(f"Mean accuracy of batch:{batch_path} is:\n {round(accuracy_mean,3)}%")
    
    return accuracy_mean

                    
                    
def test_list(bag_obj:BagReader,list_batch_path:str)->None:
    print(f"[INFO]  Testing List")
    accuracy_arr = []
    for bag_batch in list_batch_path:
        accuracy_arr.append(test_batch(bag_obj,bag_batch))
    
    accuracy_mean = np.mean(accuracy_arr)
    print(f"Mean accuracy of total is: {round(accuracy_mean,3)}%")
    
    return accuracy_mean

def main():
    os.chdir(os.path.dirname(__file__))

    bag_obj = BagReader()
    # get arguments
    parser = Parser.get_parser()
    args = Parser.get_args(parser)

    bag_file = PATH+'env_clf/bag/bag_batch_2023-06-21_16-28-26_ALL_test/2023-06-21-16-30-41.bag' # default for example and debug
    if args.bag_batch_folder is not None:
        test_batch(bag_obj,args.bag_batch_folder)

    elif args.bag_batch_list is not None:
        test_list(bag_obj,args.bag_batch_list)

    else:
        if args.single_bag is not None:
            bag_file = args.single_bag
        
        bag_obj.bag = bag_file
        accuracy = test_bag(bag_obj)
    

if __name__ == '__main__':
    main()