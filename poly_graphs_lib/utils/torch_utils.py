import torch
from torch_geometric.profile.utils import get_model_size

def get_total_dataset_bytes(dataset):
    count=0
    for data in dataset:
        for key, item in data:
            if isinstance(item,torch.Tensor):
                count+=item.element_size() * item.nelement()

    return count


def get_model_bytes(model):
    total_bytes = get_model_size(model)
    return total_bytes
