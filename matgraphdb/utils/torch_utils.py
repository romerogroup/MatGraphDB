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
