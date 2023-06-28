import os
import sys
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

def plot_test_data_amount(file_path):
    """
    Plot test accuracy against fraction from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    """
    df = pd.read_csv(file_path)
    fraction = df['fraction'].to_numpy()
    test_accuracy = df['test_accuracy'].to_numpy()

    plt.plot(fraction, test_accuracy)
    plt.scatter(fraction, test_accuracy,color='k')
    plt.xlabel('Train data in [%] of the total data')
    plt.ylabel('Test Accuracy [%]')
    plt.title('Test Accuracy vs. Amount of train data')
    
    # Set x-axis ticks with 0.05 steps
    plt.xticks(np.arange(min(fraction), max(fraction)+0.05, 0.05))
    # Set y-axis ticks with 0.01 steps
    plt.yticks(np.arange(min(test_accuracy), 1.0, 0.02))
    
    plt.show()


def main():
    """
    Main function for executing the script.

    """
    parser = argparse.ArgumentParser(description='Plot test accuracy against fraction from a CSV file.')
    parser.add_argument('--f', dest='file_path', type=str, help='Path to the CSV file.')
    parser.add_argument('--data_amount', dest='data_amount', action='store_true', default=False,
                        help='Flag indicating if the test to be executed is data_amount.')
    args = parser.parse_args()
    if args.data_amount:
        plot_test_data_amount(args.file_path)


if __name__ == '__main__':
    main()
