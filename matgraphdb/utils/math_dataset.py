import pandas as pd
import torch

def target_statistics(dataset, device):
    y_train_vals = []
    n_graphs = len(dataset)
    for data in dataset:
        data.to(device)
        y_train_vals.append(data.y)

    y_vals = torch.tensor(y_train_vals).to(device)
    avg_y_val = torch.mean(y_vals, axis=0)
    std_y_val = torch.std(y_vals, axis=0)
    return avg_y_val, std_y_val
