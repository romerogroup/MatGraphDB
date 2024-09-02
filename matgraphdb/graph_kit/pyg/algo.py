import torch
from torch_geometric.data import Data
from torch_geometric.transforms import FeaturePropagation


def feature_propagation(data=None, x=None, edge_index=None, **kwargs):
    if data is None and x is None and edge_index is None:
        raise ValueError("Either data or x and edge_index must be provided")
    if data is None:
        data = Data(x=x, edge_index=edge_index)
    else:
        x=data.x
    transform = FeaturePropagation(missing_mask=torch.isnan(x), **kwargs)
    homo_graph_transformed = transform(data)
    return homo_graph_transformed.x.numpy()
