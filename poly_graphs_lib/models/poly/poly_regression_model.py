from typing import List

import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, global_add_pool, global_mean_pool, global_max_pool, Sequential
import torch_geometric.nn as pyg_nn
from torchmetrics.functional import mean_absolute_percentage_error


class MLP(nn.Module):
    def __init__(self, input_size, layers, output_size=None):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        # Create the hidden layers
        prev_size = input_size
        for layer_size in layers:
            self.layers.append(nn.Linear(prev_size, layer_size))
            self.layers.append(nn.ReLU())
            prev_size = layer_size

        # Create the output layer
        if output_size is not None:
            self.layers.append(nn.Linear(prev_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PolyhedronRegressionModel(nn.Module):
    """This is the main Polyhedron Model. 

    Parameters
    ----------
        n_node_features : int
        The number of node features
    n_edge_features : int
        The number of edge features, by default 2
    n_gc_layers : int, optional
        The number of graph convolution layers, by default 1
    global_pooling_method : str, optional
        The global pooling method to be used, by default 'add'
    """

    def __init__(self,
                n_node_features:int,
                n_edge_features:int, 
                n_gc_layers:int=1, 
                layers_1:List[int]=[],
                layers_2:List[int]=[],
                global_pooling_method:str='add'):
        """This is the main Polyhedron Model. 

        Parameters
        ----------
         n_node_features : int
            The number of node features
        n_edge_features : int
            The number of edge features, by default 2
        n_gc_layers : int, optional
            The number of graph convolution layers, by default 1
        global_pooling_method : str, optional
            The global pooling method to be used, by default 'add'
        """
        super().__init__()

            
        layers=[]
        for i_gc_layer in range(n_gc_layers):
            if i_gc_layer == 0:
                vals = " x, edge_index, edge_attr -> x0 "
            else:
                vals = " x" + repr(i_gc_layer - 1) + " , edge_index, edge_attr -> x" + repr(i_gc_layer)

            layers.append((pyg_nn.CGConv(n_node_features, dim=n_edge_features,aggr = 'add'),vals))

        self.relu = nn.ReLU()
        self.cg_conv_layers = Sequential(" x, edge_index, edge_attr " , layers)

        if layers_1 != []:
            self.mlp_1 = MLP( input_size=n_node_features, layers=layers_1 )
        else:
            self.mlp_1 = None

        if layers_2 != []:
            if self.mlp_1:
                self.mlp_2 = MLP( input_size=layers_1[-1], layers=layers_2 )
            else:
                self.mlp_2 = MLP( input_size=n_node_features, layers=layers_2 )
        else:
            self.mlp_2 = None

        if self.mlp_2 is None and self.mlp_1 is None:
            self.out_layer= nn.Linear( n_node_features,  1)
        elif self.mlp_2 is None and self.mlp_1 is not None:
            self.out_layer= nn.Linear( layers_1[-1],  1)
        elif self.mlp_2 is not None:
            self.out_layer= nn.Linear( layers_2[-1],  1)

    
        if global_pooling_method == 'add':
            self.global_pooling_layer = global_add_pool
        elif global_pooling_method == 'mean':
            self.global_pooling_layer = global_mean_pool
        elif global_pooling_method == 'max':
            self.global_pooling_layer = global_max_pool

    def forward(self, data_batch, targets=None):
        """The forward pass of of the network

        Parameters
        ----------
        x : pygeometic.Data
            The pygeometric data object
        targets : float, optional
            The target value to use to calculate the loss, by default None

        Returns
        -------
        _type_
            _description_
        """
        
        x, edge_index, edge_attr = data_batch.x, data_batch.edge_index, data_batch.edge_attr
        batch = data_batch.batch

        x_out = x
        edge_out = edge_attr

        # Convolutional layers combine nodes and edge interactions
        out = self.cg_conv_layers(x_out, edge_index, edge_out )
        
        # First mlp
        if self.mlp_1 is not None:
            out = self.mlp_1(out)

        # Batch global pooling
        out = self.global_pooling_layer(out, batch = batch)

        # Second mlp
        if self.mlp_2 is not None:
            out = self.mlp_2(out)

        # Out layer
        out = self.out_layer(out)

        return out
    
    def encode(self, data_batch):
        x, edge_index, edge_attr = data_batch.x, data_batch.edge_index, data_batch.edge_attr
        batch = data_batch.batch

        x_out = x
        edge_out = edge_attr

        # Convolutional layers combine nodes and edge interactions
        out = self.cg_conv_layers(x_out, edge_index, edge_out ) # out -> (n_total_node_in_batch, n_node_features)

        if self.mlp_1 is not None:
            out = self.mlp_1(out)

        # Batch global pooling
        out = self.global_pooling_layer(out, batch = batch)

        # Second mlp
        if self.mlp_2 is not None:
            out = self.mlp_2(out) 
            
        return out
    
    def encode_2(self, data_batch):
        x, edge_index, edge_attr = data_batch.x, data_batch.edge_index, data_batch.edge_attr
        batch = data_batch.batch

        x_out = x
        edge_out = edge_attr

        # Convolutional layers combine nodes and edge interactions
        out = self.cg_conv_layers(x_out, edge_index, edge_out ) # out -> (n_total_node_in_batch, n_node_features)

        if self.mlp_1 is not None:
            
            out = self.mlp_1(out)
 

        # Batch global pooling
        out = self.global_pooling_layer(out, batch = batch)
        
        return out