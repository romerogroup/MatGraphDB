import os

import matplotlib.pyplot as plt
import numpy as np

def plot_multiclass_metrics(save_dir,metrics_data,metric,epochs):
    num_classes = len(metrics_data['train'][metric][0])  # Number of classes inferred from the length of the first list
    plt.figure(figsize=(10, 6))
    for class_idx in range(num_classes):
        
        train_values = [epoch[class_idx] for epoch in metrics_data['train'][metric]]
        test_values = [epoch[class_idx] for epoch in metrics_data['test'][metric]]
        
        plt.plot(epochs, train_values, label=f'Train {metric} Class {class_idx}')
        plt.plot(epochs, test_values, label=f'Test {metric} Class {class_idx}')

    plt.title(f'{metric.capitalize()}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
        
    plot_filename = os.path.join(save_dir, f"{metric}.png")
    plt.savefig(plot_filename)
    plt.close()



def plot_metrics(metrics_data, save_dir):
    """
    Plots the training and test loss curves for each metric in the given metrics data,
    and saves the plots to the specified directory.

    Args:
        metrics_data (dict): A dictionary containing 'train' and 'test' metrics data.
        save_dir (str): Path to the directory where the plots will be saved.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Create the directory if it does not exist

    # Assuming there are epochs data and the same metrics in train and test
    epochs = range(len(metrics_data['train']['accuracy']))  # Example to generate epoch data if not provided

    # Handling multiclass metrics
    multiclass_metrics = ['precision', 'recall', 'f1']
    
    for metric in metrics_data['train']:
        if metric not in ['confusion_matrix', 'epoch']:  # Exclude non-numeric metrics
            # Single value metrics like accuracy and batch_loss
            train_values = np.array(metrics_data['train'][metric])
            test_values = np.array(metrics_data['test'][metric])

            std_train=None
            std_test=None
            if metric in multiclass_metrics:
                # Compute the mean and standard deviation for each class
                std_train = np.std(train_values, axis=1)
                train_values = np.mean(train_values, axis=1)
                std_test = np.std(test_values, axis=1)
                test_values = np.mean(test_values, axis=1)
                
            # Single value metrics like accuracy and batch_loss
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_values, label=f'Train {metric}')
            plt.plot(epochs, test_values, label=f'Test {metric}')
            if std_train is not None:
                plt.errorbar(epochs, train_values, yerr=std_train, fmt='o', color='blue', ecolor='lightgray', elinewidth=3, capsize=0)
                plt.errorbar(epochs, test_values, yerr=std_test, fmt='o', color='blue', ecolor='lightgray', elinewidth=3, capsize=0)

            plt.title(metric.capitalize())
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            
            plot_filename = os.path.join(save_dir, f"{metric}_plot.png")
            plt.savefig(plot_filename)
            plt.close()

if __name__=='__main__':
    import json

    save_dir='data/production/materials_project/ML/scratch_runs/scratch'
    with open(os.path.join(save_dir,'metrics.json'), 'r') as f:
        metrics_data = json.load(f)
    plot_metrics(metrics_data,os.path.join(save_dir,'plots'))

    