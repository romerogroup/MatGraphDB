import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GraphConv,
    GraphSAGE,
    SAGEConv,
    to_hetero,
    to_hetero_with_bases,
)

from matgraphdb.pyg.models.base import FeedFoward


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


if __name__ == "__main__":
    model = GNN(hidden_channels=64, out_channels=dataset.num_classes)
    model = to_hetero(model, data.metadata(), aggr="sum")


# class HeteroInputLayer(nn.Module):
#     def __init__(self, data, n_embd: int, device="cuda:0"):
#         super().__init__()

#         self.embs = nn.ModuleDict()
#         self.data_lins = nn.ModuleDict()

#         for node_type in data.node_types:
#             num_nodes = data[node_type].num_nodes
#             num_features = data[node_type].num_node_features

#             self.embs[node_type] = nn.Embedding(num_nodes, n_embd, device=device)
#             if num_features != 0:
#                 self.data_lins[node_type] = nn.Linear(
#                     num_features, n_embd, device=device
#                 )

#     def forward(self, data):
#         x_dict = {}
#         edge_index_dict = {}
#         edge_attr_dict = {}
#         for node_type, emb_layer in self.embs.items():
#             # Handling nodes based on feature availability
#             if node_type in self.data_lins:
#                 x_dict[node_type] = self.data_lins[node_type](
#                     data[node_type].x
#                 ) + emb_layer(data[node_type].node_id)
#             else:
#                 x_dict[node_type] = emb_layer(data[node_type].node_id)

#             # edge_index_dict[node_type] = data[node_type].edge_index
#             # edge_attr_dict[node_type] = data[node_type].edge_attr

#         return x_dict


# class HeteroConvModel(nn.Module):
#     def __init__(
#         self,
#         data,
#         n_embd: int,
#         out_channels: int,
#         prediction_node_type: str,
#         n_conv_layers=1,
#         aggr="sum",
#         dropout=0.0,
#         conv_params={"dropout": 0.0, "act": "relu", "act_first": True},
#         device="cuda:0",
#     ):
#         super(HeteroConvModel, self).__init__()
#         self.prediction_node_type = prediction_node_type
#         self.input_layer = HeteroInputLayer(data, n_embd, device=device)

#         self.fwd1_dict = nn.ModuleDict()
#         for node_type in data.node_types:
#             self.fwd1_dict[node_type] = FeedFoward(n_embd, dropout=dropout)

#         # Initialize and convert GraphSAGE to heterogeneous
#         self.graph_conv = GraphSAGE(n_embd, n_embd, n_conv_layers, **conv_params)
#         self.stacked_conv = to_hetero(self.graph_conv, metadata=data.metadata())
#         # self.stacked_conv = to_hetero_with_bases(model, metadata, bases=3)

#         self.fwd2_dict = nn.ModuleDict()
#         for node_type in data.node_types:
#             self.fwd2_dict[node_type] = FeedFoward(n_embd, dropout=dropout)

#         self.output_layer = nn.Linear(n_embd, out_channels)

#     def forward(self, data):
#         x_dict = self.input_layer(data)
#         for node_type in data.node_types:
#             x_dict[node_type] = self.fwd1_dict[node_type](x_dict[node_type])
#         x_dict = self.stacked_conv(x_dict, data.edge_index_dict)
#         for node_type in data.node_types:
#             x_dict[node_type] = self.fwd2_dict[node_type](x_dict[node_type])

#         out = self.output_layer(x_dict[self.prediction_node_type])
#         return out
