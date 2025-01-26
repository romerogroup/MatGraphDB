import torch
import torch.nn as nn
from torch_geometric.nn import CGConv

from matgraphdb.pyg.models.base import InputLayer, MultiLayerPercetronLayer


class CGConvBlock(torch.nn.Module):
    def __init__(self, input_channels, num_edge_features):
        super(CGConvBlock, self).__init__()
        self.num_edge_features = num_edge_features
        self.conv = CGConv(input_channels, dim=num_edge_features)
        self.ffw = MultiLayerPercetronLayer(input_channels, input_channels)

    def forward(self, x, edge_index, edge_attr):
        if self.num_edge_features > 0:
            x = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
        else:
            x = self.conv(x=x, edge_index=edge_index)
        x = self.ffw(x)
        return x


class CrystalGraphEncoder(nn.Module):
    def __init__(
        self, num_node_features, num_edge_features, hidden_channels, num_layers=1
    ):
        super().__init__()
        self.conv_blocks = nn.ModuleList(
            [CGConvBlock(hidden_channels, num_edge_features) for _ in range(num_layers)]
        )
        self.proj = nn.Linear(num_node_features, hidden_channels)

    def forward(self, data):
        for conv_block in self.conv_blocks:
            x = conv_block(data.x, data.edge_index, data.edge_attr)
        x = self.proj(x)
        return x


if __name__ == "__main__":
    from torch_geometric.data import Data

    data = Data()
    data.edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data.edge_attr = torch.tensor([[0.5], [0.5], [0.5], [0.5]])
    data.x = torch.tensor([[0.0], [0.0], [0.0], [0.0]])
    print(data)

    model = CrystalGraphEncoder(
        num_node_features=1, num_edge_features=1, hidden_channels=1
    )
    print(model(data))
