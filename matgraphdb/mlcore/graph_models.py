# Creating a GraphSAGE model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid

# Define the model
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GraphSAGE, self).__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.convs = nn.ModuleList([
            SAGEConv(hidden_channels, hidden_channels)
            for _ in range(num_layers - 2)
        ])
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, adjs):
        x = F.relu(self.conv1(x, adjs[0]))
        x = F.dropout(x, p=0.5, training=self.training)
        for conv in self.convs:
            x = F.relu(conv(x, adjs[i + 1]))
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, adjs[-1])
        return F.log_softmax(x, dim=-1)

