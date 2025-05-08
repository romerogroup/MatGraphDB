from typing import Dict, Union

import torch
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch_geometric import nn as pyg_nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.nn import HANConv
from matgraphdb.pyg.models.base import MultiLayerPercetronLayer


class HAN(torch.nn.Module):
    def __init__(self, 
                 in_channels: Union[int, Dict[str, int]],
                 out_channels: int,
                 data: HeteroData,
                 out_node_name: str='materials',
                 hidden_channels=128, 
                 dropout=0.6,
                 heads=8):
        super().__init__()
        self.out_node_name = out_node_name
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads,
                                dropout=dropout, metadata=data.metadata())
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        out = self.lin(out[self.out_node_name])
        return out