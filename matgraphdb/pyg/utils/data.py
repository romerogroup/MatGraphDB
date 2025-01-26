import torch
from torch_geometric.loader import DataLoader


def split_data(
    data_list,
    train_size_ratio=0.8,
    val_ratio=0.1,
    batch_size=32,
    seed=None
):
    """
    Splits data into train/val/test and creates DataLoaders
    Returns: (train_loader, val_loader, test_loader)
    """
    if seed is not None:
        torch.manual_seed(seed)

    total_size = len(data_list)
    train_size = int(train_size_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Shuffle the dataset
    indices = torch.randperm(total_size).tolist()
    train_data = [data_list[i] for i in indices[:train_size]]
    val_data = [data_list[i] for i in indices[train_size:train_size+val_size]]
    test_data = [data_list[i] for i in indices[train_size+val_size:]]

    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(val_data, batch_size=batch_size),
        DataLoader(test_data, batch_size=batch_size)
    )