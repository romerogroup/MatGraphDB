import torch
import torch.nn as nn
from torch_geometric.nn import CGConv

from matgraphdb.pyg.models.base import InputLayer, MultiLayerPercetronLayer
from matgraphdb.pyg.models.hetero_crystal_cl.crystal_graph_encoder import (
    CrystalGraphEncoder,
)
from matgraphdb.pyg.models.hetero_crystal_cl.hetero_encoder import HeteroEncoder


class HeteroCrystalCL(nn.Module):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        hidden_channels,
        crystal_graph_kwargs=None,
        hetero_encoder_kwargs=None,
    ):
        if crystal_graph_kwargs is None:
            crystal_graph_kwargs = {}
        if hetero_encoder_kwargs is None:
            hetero_encoder_kwargs = {}

        super().__init__()
        self.input_layer = InputLayer(num_node_features, hidden_channels)

        self.hetero_encoder = HeteroEncoder(
            num_node_features,
            num_edge_features,
            hidden_channels,
            **hetero_encoder_kwargs,
        )

        self.crystal_graph_encoder = CrystalGraphEncoder(
            num_node_features,
            num_edge_features,
            hidden_channels,
            **crystal_graph_kwargs,
        )

        self.ffw = MultiLayerPercetronLayer(hidden_channels, hidden_channels)

    def forward(self, hetero_data, graph_data):
        hetero_x = self.hetero_encoder(hetero_data)
        graph_x = self.crystal_graph_encoder(graph_data)

        hetero_z = self.ffw(hetero_x)
        graph_z = self.ffw(graph_x)

        return hetero_z, graph_z
