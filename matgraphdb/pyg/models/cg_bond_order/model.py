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


class DecoderBlock(nn.Module):
    def __init__(self, hidden_channels, num_edge_features, num_layers=1):
        super().__init__()
        self.num_edge_features = num_edge_features

        self.ffw_layer = MultiLayerPercetronLayer(
            2 * hidden_channels + num_edge_features,
            2 * hidden_channels + num_edge_features,
        )
        self.node_conv_layer = CGConv(hidden_channels, dim=num_edge_features)

    def forward(self, z, edge_attr, edge_index):

        src, dst = edge_index

        # Get node features for source and target nodes
        z_src = z[src]  # Shape: (num_edges, hidden_channels)
        z_dst = z[dst]  # Shape: (num_edges, hidden_channels)

        if self.num_edge_features > 0:
            z_edge = torch.cat(
                [z_src, z_dst, edge_attr], dim=-1
            )  # Shape: (num_edges, 2*hidden_channels + num_edge_features)
        else:
            z_edge = torch.cat([z_src, z_dst], dim=-1)
        z_edge = self.ffw_layer(z_edge)

        z_node = self.node_conv_layer(z, edge_index)
        return z_node, z_edge


class Decoder(nn.Module):
    def __init__(self, num_node_features, output_dim, num_edge_features, num_layers=1):
        super().__init__()
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.decoder_blocks.append(
                DecoderBlock(num_node_features, num_edge_features)
            )
        self.edge_out_layer = nn.Linear(2 * num_node_features, output_dim)

    def forward(self, z, edge_attr, edge_index):
        # Get source and target node indices from edge_index
        z_node, z_edge = z, edge_attr
        for block in self.decoder_blocks:
            z_node, z_edge = block(z_node, z_edge, edge_index)
        z_edge = self.edge_out_layer(z_edge)
        return z_node, z_edge


# class BondOrderPredictor(nn.Module):
#     def __init__(
#         self,
#         num_node_features,
#         num_edge_features,
#         hidden_channels,
#         output_dim,
#         num_layers=2,
#     ):
#         super().__init__()
#         self.encoder = Encoder(
#             num_node_features,
#             num_edge_features,
#             hidden_channels,
#             num_layers,
#         )
#         # self.decoder = Decoder(
#         #     num_node_features=hidden_channels,
#         #     output_dim=output_dim,
#         #     num_edge_features=num_edge_features,
#         #     num_layers=num_layers,
#         # )

#         self.decoder_blocks = nn.ModuleList()
#         for i in range(num_layers):
#             self.decoder_blocks.append(
#                 MultiLayerPercetronLayer(2*num_node_features, 2*num_node_features)
#             )


#     def forward(self, data):
#         z = self.encoder(data.x, data.edge_index, data.edge_attr)  # (B, N, C)

#         node_attr, edge_attr = self.decoder(z, data.edge_attr, data.edge_index)


#         src, dst = data.edge_index

#         # Get node features for source and target nodes
#         z_src = z[src]  # Shape: (num_edges, hidden_channels)
#         z_dst = z[dst]  # Shape: (num_edges, hidden_channels)
#         return node_attr, edge_attr


class BondOrderPredictor(nn.Module):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        hidden_channels,
        output_dim,
        num_layers=2,
    ):
        super().__init__()

        self.num_edge_features = num_edge_features
        self.encoder = Encoder(
            num_node_features,
            num_edge_features,
            hidden_channels,
            num_layers,
        )

        self.ffw_layers = nn.ModuleList()
        for i in range(num_layers):
            self.ffw_layers.append(
                MultiLayerPercetronLayer(2 * num_node_features, 2 * num_node_features)
            )
        self.out_layer = nn.Linear(2 * num_node_features, output_dim)

    def forward(self, data):
        z = self.encoder(data.x, data.edge_index, data.edge_attr)  # (B, N, C)

        # node_attr, edge_attr = self.decoder(z, data.edge_attr, data.edge_index)

        src, dst = data.edge_index

        # Get node features for source and target nodes
        z_src = z[src]  # Shape: (num_edges, hidden_channels)
        z_dst = z[dst]  # Shape: (num_edges, hidden_channels)

        if self.num_edge_features > 0:
            edge_attr = torch.cat([z_src, z_dst, data.edge_attr], dim=-1)
        else:
            edge_attr = torch.cat([z_src, z_dst], dim=-1)

        for ffw_layer in self.ffw_layers:
            edge_attr = ffw_layer(edge_attr)

        edge_attr = self.out_layer(edge_attr)
        return z, edge_attr
