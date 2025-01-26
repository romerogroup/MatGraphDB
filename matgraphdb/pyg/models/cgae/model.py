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


class Encoder(nn.Module):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        hidden_channels,
        num_layers=1,
    ):
        super().__init__()
        self.input_layer = InputLayer(num_node_features, hidden_channels)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(CGConvBlock(hidden_channels, num_edge_features))

    def forward(self, x, edge_index, edge_attr):

        x = self.input_layer(x)

        for block in self.blocks:

            x = block(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return x


# class Decoder(nn.Module):
#     def __init__(self, hidden_channels, output_dim, num_edge_features, num_layers=1):
#         super().__init__()
#         self.blocks = nn.ModuleList()
#         for i in range(num_layers):
#             self.blocks.append(CGConvBlock(hidden_channels, num_edge_features))

#         self.out_layer = nn.Linear(hidden_channels, output_dim)

#     def forward(self, x, edge_index, edge_attr, batch):
#         for block in self.blocks:
#             x = block(x=x, edge_index=edge_index, edge_attr=edge_attr)
#         return self.out_layer(x)


class Decoder(nn.Module):
    def __init__(self, n_embd, n_node_features, n_edge_features, num_layers=1):
        super(Decoder, self).__init__()

        # self.ffw = MultiLayerPercetronLayer(n_embd, n_embd)
        self.ffw_layers = nn.ModuleList()
        for i in range(num_layers):
            self.ffw_layers.append(MultiLayerPercetronLayer(n_embd, n_embd))
        self.node_decoder = nn.Linear(n_embd, n_node_features)
        self.edge_attr_decoder = nn.Linear(n_embd, n_edge_features)

    def forward(self, z, batch_indices):

        graphs = pyg_utils.unbatch(z, batch_indices)

        x_list = []
        edge_attr_list = []
        for graph_z in graphs:
            node_z, edge_attr_z = self.decode_graph(graph_z)

            x_list.append(node_z)
            edge_attr_list.append(edge_attr_z)

        x = torch.cat(x_list, dim=0)
        edge_attr = torch.cat(edge_attr_list, dim=0)
        return x, edge_attr

    def decode_graph(self, graph_z):

        for ffw_layer in self.ffw_layers:
            graph_z = ffw_layer(graph_z)

        # A_hat = torch.sigmoid(torch.matmul(graph_z, graph_z.t()))
        # return A_hat
        node_z = self.node_decoder(graph_z).relu()
        edge_attr_z = self.edge_attr_decoder(graph_z).relu()
        return node_z, edge_attr_z


class CGAE(nn.Module):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        hidden_channels,
        num_layers=2,
    ):
        super().__init__()
        self.encoder = Encoder(
            num_node_features,
            num_edge_features,
            hidden_channels,
            num_layers,
        )
        self.decoder = Decoder(hidden_channels, num_node_features, num_edge_features)

    def forward(self, data):
        z = self.encoder(data.x, data.edge_index, data.edge_attr)

        x, edge_attr = self.decoder(z, data.edge_indexdata.batch)
        return x, edge_attr, z
