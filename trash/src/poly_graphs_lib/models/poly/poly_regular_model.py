from typing import List

import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, global_add_pool, global_mean_pool, global_max_pool, Sequential
import torch_geometric.nn as pyg_nn


class PolyhedronRegularModel(nn.Module):
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
                n_fc_layers:int=1,
                n_neurons=None,
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
        self.n_node_features=n_node_features
        self.n_edge_features=n_edge_features
        self.n_fc_layers=n_fc_layers
        self.n_gc_layers=n_gc_layers
        self.n_neurons=n_neurons
            

        self.activation = nn.ReLU()


        gc_layers = self._add_gc_layers()
        self.cg_conv_layers = Sequential(" x, edge_index, edge_attr " , gc_layers)

        fc_layers=self._add_fc_layers()
        self.fc_layers = Sequential(" x" + repr(self.n_gc_layers - 1)  , fc_layers)
        

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
        
        # Fully connected layers
        out = self.fc_layers(out)

        # Batch global pooling
        out = self.global_pooling_layer(out, batch = batch)

        return out
    
    def encode_2(self, data_batch):
        x, edge_index, edge_attr = data_batch.x, data_batch.edge_index, data_batch.edge_attr
        batch = data_batch.batch

        x_out = x
        edge_out = edge_attr

        # Convolutional layers combine nodes and edge interactions
        out = self.cg_conv_layers(x_out, edge_index, edge_out ) # out -> (n_total_node_in_batch, n_node_features)

        # Convolutional layers combine nodes and edge interactions
        out = self.cg_conv_layers(x_out, edge_index, edge_out )
        
        # Fully connected layers
        out = self.fc_layers(out)

        # Batch global pooling
        out = self.global_pooling_layer(out, batch = batch)
            
        return out
    
    def _add_gc_layers(self):
        gc_layers=[]
        for i_gc_layer in range(self.n_gc_layers):
            if i_gc_layer == 0:
                vals = " x, edge_index, edge_attr -> x0 "
            else:
                vals = " x" + repr(i_gc_layer - 1) + " , edge_index, edge_attr -> x" + repr(i_gc_layer)
            gc_layers.append((pyg_nn.CGConv(self.n_node_features, dim=self.n_edge_features,aggr = 'add'),vals))
        return gc_layers
        
    def _add_fc_layers(self):
        fc_layers = []
        if self.n_neurons is None:
            self.n_neurons = self.n_node_features

        if self.n_fc_layers == 1:  # if only one fully connected layer

            if self.n_gc_layers > 0:

                vals = " x" + repr(self.n_gc_layers - 1) + " -> y "

            else:

                vals = " x -> y "

            new_layer = (torch.nn.Linear(self.n_node_features, 1), vals)
            fc_layers.append(new_layer)

        else:

            if self.n_gc_layers > 0:

                vals = " x" + repr(self.n_gc_layers - 1) + " -> y0 "

            else:

                vals = " x -> y0 "

            new_layer = (torch.nn.Linear(self.n_node_features, self.n_neurons), vals)
            fc_layers.append(new_layer)

            vals = " y0 -> y0 "
            new_activation = (self.activation, vals)

            fc_layers.append(new_activation)

            for n_layer in range(1, self.n_fc_layers - 1):

                vals = " y" + repr(n_layer - 1) + " -> y" + repr(n_layer)
                new_layer = (torch.nn.Linear(self.n_neurons, self.n_neurons), vals)
                fc_layers.append(new_layer)

                vals = "y" + repr(n_layer) + " -> y" + repr(n_layer)
                new_activation = (self.activation, vals)
                fc_layers.append(new_activation)

            # and finally the exit layer

            vals = " y" + repr(self.n_fc_layers - 2) + " -> y "
            new_layer = (torch.nn.Linear(self.n_neurons, 1), vals)
            fc_layers.append(new_layer)
        return fc_layers