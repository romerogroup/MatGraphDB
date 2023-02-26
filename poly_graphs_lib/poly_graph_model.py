from typing import List

import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, global_add_pool, global_mean_pool, global_max_pool, Sequential
import torch_geometric.nn as pyg_nn
from torchmetrics.functional import mean_absolute_percentage_error


class PolyhedronModel(nn.Module):
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
                n_hidden_layers:List[int]=5,
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

            layers.append((pyg_nn.CGConv(n_node_features, dim=n_edge_features),vals))

        # self.cg_conv_layers = Sequential(" x, edge_index, edge_attr, batch " , layers)
        self.cg_conv_layers = Sequential(" x, edge_index, edge_attr " , layers)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.linear_1 = nn.Linear( n_node_features, n_hidden_layers)
        self.out_layer= nn.Linear( n_hidden_layers[-1],  1)

        if global_pooling_method == 'add':
            self.global_pooling_layer = global_add_pool
        elif global_pooling_method == 'mean':
            self.global_pooling_layer = global_mean_pool
        elif global_pooling_method == 'max':
            self.global_pooling_layer = global_max_pool


    def forward(self, x, targets=None):
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
        # Convolutional layers combine nodes and edge interactions
        out = self.cg_conv_layers(x.x, x.edge_index, x.edge_attr ) # out -> (n_total_node_in_batch, n_node_features)
        out = self.sig(out) # out -> (n_total_nodes_in_batch, n_node_features)

        # Fully connected layer
        out = self.linear_1(out) # out -> (n_total_nodes_in_batch, n_hidden_layers[0])
        out = self.sig(out) # out -> (n_total_nodes_in_batch, n_hidden_layers[0])

        # batch is index list differteriating which nodes belong to which graph
        out = self.global_pooling_layer(out, batch = x.batch) # out -> (n_graphs, n_hidden_layers[0])

        out = self.out_layer(out) # out -> (n_graphs, 1)
        out = self.relu(out) # out -> (n_graphs, 1)

        # Loss handling
        if targets is None:
            loss = None
            mape_loss = None
        else:
            loss_fn = torch.nn.MSELoss()
            mape_loss = mean_absolute_percentage_error(torch.squeeze(out, dim=1), targets)
            loss = loss_fn(torch.squeeze(out, dim=1), targets)

        return out,  loss, mape_loss

    def generate_encoding(self, x):
        """This method generates the polyhedra encoding

        Parameters
        ----------
        x : pyg.Data object
            The pygeometric Data object

        Returns
        -------
        torch.Tensor
            The encoded polyhedra vector
        """
        out = self.cg_conv_layers(x.x, x.edge_index, x.edge_attr )
        out = self.relu(out)
        out = self.linear_1(out) # out -> (n_total_atoms_in_batch, 1)
        out = self.sig(out)
        out = self.global_pooling_layer(out, batch = x.batch)
        return out

