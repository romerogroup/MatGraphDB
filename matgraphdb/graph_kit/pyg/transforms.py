import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData

def min_max_normalize(tensor:torch.Tensor, normalization_range=(0,1), tensor_min=None, tensor_max=None):
    if tensor_min is None:
        tensor_min = tensor.min()
    if tensor_max is None:
        tensor_max = tensor.max()


    values_minus_min = tensor - tensor_min
    old_range_diff = tensor_max - tensor_min
    new_range_diff = normalization_range[1] - normalization_range[0]
    new_min = normalization_range[0]
    return ( values_minus_min / old_range_diff ) * new_range_diff + new_min, tensor_min, tensor_max
    
def standardize_tensor(tensor, epsilon=1e-8, mean=None, std=None):
    if mean is None:
        mean = torch.mean(tensor)
    if std is None:
        std = torch.std(tensor)
    return (tensor - mean) / (std + epsilon), mean, std


def robust_scale(tensor, q_min=0.25, q_max=0.75, epsilon=1e-8, median=None, iqr=None):
    q1 = torch.quantile(tensor, q_min)
    q3 = torch.quantile(tensor, q_max)
    if iqr is None:
        iqr = q3 - q1
    if median is None:
        median = torch.median(tensor)
    return (tensor - median) / (iqr + epsilon), median, iqr