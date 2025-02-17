import torch
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch_geometric import nn as pyg_nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero

from matgraphdb.pyg.models.base import MultiLayerPercetronLayer


class GNNEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        out_channels=None,
        num_conv_layers=1,
        num_ffw_layers=1,
        dropout=0.0,
    ):
        super().__init__()

        self.num_conv_layers = num_conv_layers
        self.num_ffw_layers = num_ffw_layers
        self.out_channels = out_channels

        # Create ModuleLists to store layers
        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.ffw_lins = torch.nn.ModuleList()
        self.ffw_batch_norms = torch.nn.ModuleList()

        self.input_conv = pyg_nn.SAGEConv((-1, -1), hidden_channels, aggr="mean")
        for _ in range(num_conv_layers):
            self.convs.append(pyg_nn.SAGEConv((-1, -1), hidden_channels, aggr="mean"))
            self.lins.append(pyg_nn.Linear(-1, hidden_channels))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(num_ffw_layers):
            self.ffw_lins.append(pyg_nn.Linear(-1, hidden_channels))
            self.ffw_batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))

        # Last layer
        if self.out_channels is not None:
            self.out_conv = pyg_nn.SAGEConv((-1, -1), out_channels, aggr="mean")
            self.proj = pyg.nn.Linear(-1, out_channels)

        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()

    def forward(self, x, edge_index):
        for i in range(self.num_conv_layers):
            x = self.convs[i](x, edge_index).relu()
            # x = self.activation(x)
            # x = self.lins[i](x).relu()
            # x = self.activation(x)
            # x = self.batch_norms[i](x)
            x = self.dropout(x)

        for i in range(self.num_ffw_layers):
            x = self.ffw_lins[i](x).relu()
            # x = self.ffw_batch_norms[i](x)
            x = self.dropout(x)

        if self.out_channels is not None:
            x = self.proj(x)

        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, src_node_name, dst_node_name, num_layers=1, dropout=0.0):
        super().__init__()
        self.src_node_name = src_node_name
        self.dst_node_name = dst_node_name
        self.mlp = MultiLayerPercetronLayer(
            input_dim=2 * hidden_channels, num_layers=num_layers, dropout=dropout
        )
        self.mlp_out = torch.nn.Linear(2 * hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict[self.src_node_name][row], z_dict[self.dst_node_name][col]], dim=-1)

        z = self.mlp(z)
        z = self.mlp_out(z)
        return z.view(-1).sigmoid()

class MaterialEdgePredictor(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        data=None,
        decoder_kwargs=None,
        encoder_kwargs=None,
        use_shallow_embedding_for_materials=False,
        use_projections=False,
        use_embeddings=False,
    ):
        """
        One of `use_projections` or `use_embeddings` must be True.
        
        :param data: A PyG heterogeneous data object.
                     data.metadata() should return a list of tuples (src_node, rel, dst_node).
        """
        super().__init__()
        if not use_projections and not use_embeddings:
            raise ValueError("Either use_projections or use_embeddings must be True")

        # Get edge metadata as a list of (src, rel, dst) tuples.
        self.node_types = data.metadata()[0]  # e.g. [("materials", "interacts", "chemicals"), ...]
        self.edge_types = data.metadata()[1]  # e.g. [("materials", "interacts", "chemicals"), ...]

        # Derive the node types from the edge types.
        node_types = set()
        for src, rel, dst in self.edge_types:
            node_types.add(src)
            node_types.add(dst)
        self.node_types = list(node_types)

        self.use_shallow_embedding_for_materials = use_shallow_embedding_for_materials
        self.use_projections = use_projections
        self.use_embeddings = use_embeddings

        # Build node embeddings or projections per node type.
        if use_embeddings:
            node_embeddings = {}
            for node_type in self.node_types:
                # If you want to avoid learning embeddings for materials using shallow embeddings:
                if not self.use_shallow_embedding_for_materials and node_type == "materials":
                    continue
                node_embeddings[node_type] = torch.nn.Embedding(
                    data[node_type].num_nodes, hidden_channels
                )
            self.node_embeddings = torch.nn.ModuleDict(node_embeddings)
        if use_projections:
            node_projections = {}
            for node_type in self.node_types:
                if hasattr(data[node_type], "x"):
                    node_projections[node_type] = torch.nn.Linear(
                        data[node_type].x.shape[1], hidden_channels
                    )
            self.node_projections = torch.nn.ModuleDict(node_projections)

        if decoder_kwargs is None:
            decoder_kwargs = {}
        if encoder_kwargs is None:
            encoder_kwargs = {}

        self.encoder = GNNEncoder(hidden_channels, hidden_channels, **encoder_kwargs)
        # Create metadata tuple required by to_hetero: (list of node types, list of edge types)
        metadata = (self.node_types, self.edge_types)
        self.encoder = pyg_nn.to_hetero(self.encoder, metadata, aggr="sum")

        # Create an EdgeDecoder for each edge type that involves 'materials'
        self.edge_types_to_decoder_keys = {}
        self.decoder_keys_to_edge_types = {}
        self.decoders = torch.nn.ModuleDict()
        for src, rel, dst in self.edge_types:
            if src == "materials" or dst == "materials":
                # Create a unique key for this edge type.
                key = f"{src}__{rel}__{dst}"
                self.edge_types_to_decoder_keys[(src, rel, dst)] = key
                self.decoder_keys_to_edge_types[key] = (src, rel, dst)
                self.decoders[key] = EdgeDecoder(
                    hidden_channels=hidden_channels, 
                    src_node_name=src, 
                    dst_node_name=dst, 
                    **decoder_kwargs
                )

    def process_input_nodes(self, hetero_data):
        
        z_dict = {}
        for node_type in self.node_types:
            x=None
            if hasattr(hetero_data[node_type], "x"):
                x=hetero_data[node_type].x
            
    
            node_ids=hetero_data[node_type].node_ids
            if self.use_projections and self.use_embeddings:
                if x is not None:
                    z_dict[node_type] = self.node_projections[node_type](x)
                
                if node_ids is None:
                    raise ValueError(
                        "node_ids must be provided if use_embeddings is True"
                    )
                if (
                    not self.use_shallow_embedding_for_materials
                    and node_type == "materials"
                ):
                    continue
                if x is not None:
                    z_dict[node_type] += self.node_embeddings[node_type](node_ids)
                else:
                    z_dict[node_type] = self.node_embeddings[node_type](node_ids)
                
            elif self.use_projections and not self.use_embeddings:
                # hetero_data[node_type].z = self.node_projections[node_type](x)
                z_dict[node_type] = self.node_projections[node_type](x)
                
                
                
            elif not self.use_projections and self.use_embeddings:
                if node_ids is None:
                    raise ValueError(
                        "node_ids must be provided if use_embeddings is True"
                    )
                if (
                    not self.use_shallow_embedding_for_materials
                    and node_type == "materials"
                ):
                    raise ValueError(
                        "use_shallow_embedding_for_materials must be False if use_embeddings is True and use_projections is False"
                    )
                # hetero_data[node_type].z = self.node_embeddings[node_type](node_ids)
                z_dict[node_type] = self.node_embeddings[node_type](node_ids)
            else:
                raise ValueError(
                    "Either use_projections or use_embeddings must be True"
                )
        return z_dict

    def forward(self, data_batch):
        """
        :param x_dict: Dictionary of node features per node type.
        :param edge_index_dict: Dictionary of edge indices for the heterogeneous graph.
        :param edge_label_index_dict: Dictionary mapping keys (e.g. "materials__rel__chemicals")
                                      to an edge index tensor (tuple of (row, col)) for that edge type.
        :param node_ids: (Optional) Dictionary mapping node types to node IDs, if using embeddings.
        :return: A dictionary mapping each decoder key to its predictions.
        """
        
        edge_index_dict = data_batch.edge_index_dict
        # Process input nodes with projections and/or embeddings.
        z_dict = self.process_input_nodes(data_batch)
        # print(z_data_batch)
        # Encode all nodes via the heterogeneous encoder.
        z_dict = self.encoder(z_dict, edge_index_dict)

        out = {}
        # Run each decoder for which an edge index is provided.
        for edge_type, key in self.edge_types_to_decoder_keys.items():
            src, rel, dst = edge_type
            if not hasattr(data_batch[src, rel, dst], "edge_label_index"):
                continue
            edge_label_index = data_batch[src, rel, dst].edge_label_index
            out[key] = self.decoders[key](z_dict, edge_label_index)
        return out

    def encode(self, data_batch):
        edge_index_dict = data_batch.edge_index_dict
        # Process input nodes with projections and/or embeddings.
        z_dict = self.process_input_nodes(data_batch)
        # print(z_data_batch)
        # Encode all nodes via the heterogeneous encoder.
        z_dict = self.encoder(z_dict, edge_index_dict)
        return z_dict