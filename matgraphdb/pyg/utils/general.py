import logging

import torch
from torch_geometric.profile.utils import get_model_size

logger = logging.getLogger(__name__)


def get_total_dataset_bytes(dataset):
    """
    Calculates the total number of bytes occupied by a dataset.

    Args:
        dataset (Iterable): The dataset to calculate the size of.

    Returns:
        int: The total number of bytes occupied by the dataset.
    """
    count = 0
    for data in dataset:
        for key, item in data:
            if isinstance(item, torch.Tensor):
                count += item.element_size() * item.nelement()

    return count


def get_model_bytes(model):
    """
    Calculates the size of a PyTorch model in bytes.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: The size of the model in bytes.
    """
    total_bytes = get_model_size(model)
    return total_bytes


def target_statistics(dataset, device):
    """
    Calculate the average and standard deviation of the target values in a dataset.

    Args:
        dataset (list): A list of data instances.
        device (torch.device): The device to perform the calculations on.

    Returns:
        tuple: A tuple containing the average and standard deviation of the target values.
    """
    y_train_vals = []
    n_graphs = len(dataset)
    for data in dataset:
        data.to(device)
        y_train_vals.append(data.y)

    y_vals = torch.tensor(y_train_vals).to(device)
    avg_y_val = torch.mean(y_vals, axis=0)
    std_y_val = torch.std(y_vals, axis=0)
    return avg_y_val, std_y_val
