import os
import copy
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import mlflow

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import CGConv, global_add_pool, global_mean_pool, global_max_pool, Sequential
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn
from torchmetrics.functional import mean_absolute_percentage_error
from torchmetrics import MeanAbsolutePercentageError

from poly_graphs_lib.pyg_dataset import PolyhedraDataset
from poly_graphs_lib.poly_regression_model import PolyhedronModel

np.set_printoptions(suppress=True)

def cosine_similarity(x,y):
    return x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))

def distance_similarity(x,y):
    return np.linalg.norm(x/np.linalg.norm(x) -y/np.linalg.norm(x))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
                n_hidden_layers:List[int]=[5],
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

        # self.cg_conv_layers = Sequential(" x, edge_index, edge_attr, batch " , layers)
        self.bn_node = pyg_nn.norm.BatchNorm(in_channels=n_node_features)
        self.bn_edge = pyg_nn.norm.BatchNorm(in_channels=n_edge_features)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)


        self.cg_conv_layers = Sequential(" x, edge_index, edge_attr " , layers)
        self.linear_1 = nn.Linear( n_node_features, n_hidden_layers[0])
        self.linear_2 = nn.Linear( n_hidden_layers[0], n_hidden_layers[-1])

        # self.linear_1 = nn.Linear( n_node_features, n_hidden_layers[0])
        # self.linear_2 = nn.Linear( n_hidden_layers[0], n_hidden_layers[1])

        self.out_layer= nn.Linear( n_hidden_layers[-1],  1)
        

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


        # x_out = self.bn_node(x)
        # edge_out = self.bn_edge(edge_attr)

        x_out = x
        edge_out = edge_attr

        # Convolutional layers combine nodes and edge interactions
        out = self.cg_conv_layers(x_out, edge_index, edge_out ) # out -> (n_total_node_in_batch, n_node_features)
        
        out = self.leaky_relu(out)
    
        
        # Fully connected layer
        out = self.linear_1(out) # out -> (n_total_nodes_in_batch, n_hidden_layers[0])
        out = self.leaky_relu(out) # out -> (n_total_nodes_in_batch, n_hidden_layers[0])
        
        # batch is index list differteriating which nodes belong to which graph
        out = self.global_pooling_layer(out, batch = batch) # out -> (n_graphs, n_hidden_layers[0])
        
        out = self.linear_2(out) # out -> (n_total_nodes_in_batch, n_hidden_layers[0])
        out = self.leaky_relu(out) 

        # # Fully connected layer
        # out = self.linear_1(out) # out -> (n_total_nodes_in_batch, n_hidden_layers[0])
        # out = self.leaky_relu(out) # out -> (n_total_nodes_in_batch, n_hidden_layers[0])

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
        out = self.leaky_relu(out)
        out = self.linear_1(out) # out -> (n_total_atoms_in_batch, 1)
        out = self.leaky_relu(out)
        out = self.global_pooling_layer(out, batch = x.batch) # out -> (n_graphs, n_hidden_layers[0])
    
        out = self.linear_2(out) # out -> (n_total_nodes_in_batch, n_hidden_layers[0])
        out = self.leaky_relu(out) 

        return out
    
    
def main():

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    batch_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate = 0.001

    # polyhedron model parameters
    n_gc_layers = 2
    n_hidden_layers=[8,4]
    global_pooling_method = 'add'

    # dataset parameters
    dataset = 'material_polyhedra'
    feature_set_index = 3
    y_val = ['energy_per_verts','dihedral_energy'][0]
    dataset = 'material_polyhedra'

    load_checkpoint = project_dir + os.sep+ 'models' + os.sep + 'model_checkpoint.pth'


    ######################################################################################
    # Testing model
    ######################################################################################
    test_dir = f"{project_dir}{os.sep}datasets{os.sep}{dataset}{os.sep}feature_set_{feature_set_index}{os.sep}test"
    train_dir = f"{project_dir}{os.sep}datasets{os.sep}{dataset}{os.sep}feature_set_{feature_set_index}{os.sep}train"


    train_dataset = PolyhedraDataset(database_dir=train_dir, device=device, y_val=y_val)
    n_node_features = train_dataset[0].x.shape[1]
    n_edge_features = train_dataset[0].edge_attr.shape[1]

    node_max = torch.zeros(n_node_features, device = device)
    node_min = torch.zeros(n_node_features, device = device)
    edge_max = torch.zeros(n_edge_features, device = device)
    edge_min = torch.zeros(n_edge_features, device = device)
    for data in train_dataset:

        # Finding node max and min
        current_node_max = data.x.max(axis=0)[0]
        node_max = torch.vstack( [node_max,current_node_max] )
        node_max = node_max.max(axis=0)[0]

        current_node_min = data.x.min(axis=0)[0]
        node_min = torch.vstack( [node_min,current_node_min] )
        node_min = node_min.min(axis=0)[0]

        # Finding edge max and min
        current_edge_max = data.edge_attr.max(axis=0)[0]
        edge_max = torch.vstack( [edge_max,current_edge_max] )
        edge_max = edge_max.max(axis=0)[0]

        current_edge_min = data.edge_attr.min(axis=0)[0]
        edge_min = torch.vstack( [edge_min,current_edge_min] )
        edge_min = edge_min.min(axis=0)[0]

    del train_dataset

    def min_max_scaler(data):
        data.x = ( data.x - node_min ) / (node_min - node_max)
        data.edge_attr = ( data.edge_attr - edge_min ) / (edge_min - edge_max)

        data.edge_attr=data.edge_attr.nan_to_num()
        data.x=data.x.nan_to_num()
        return data


    test_dataset = PolyhedraDataset(database_dir=test_dir, device=device, y_val=y_val, transform = min_max_scaler)

    # Creating data loaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    loaded_checkpoint = torch.load(load_checkpoint)
        
    model = PolyhedronModel(n_node_features=n_node_features, 
                                n_edge_features=n_edge_features, 
                                n_gc_layers=n_gc_layers,
                                n_hidden_layers=n_hidden_layers,
                                global_pooling_method=global_pooling_method)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.load_state_dict(loaded_checkpoint['model_state'])
    optimizer.load_state_dict(loaded_checkpoint['optim_state'])
    n_epoch = loaded_checkpoint['epoch']
    train_loss = loaded_checkpoint['train_loss']
    val_loss = loaded_checkpoint['val_loss']
    print(repr(n_epoch) + ",  " + repr(train_loss) + ",  " + repr(val_loss))
    data = []
    for sample in test_loader:
        predictions = model(sample)
        # print(predictions)
        # print(predictions.shape)
        for real, pred, encoding in zip(sample.y,predictions[0],model.generate_encoding(sample)):
        #     print('______________________________________________________')
            print(f"Prediction : {pred.item()} | Expected : {real.item()} | Percent error : { abs(real.item() - pred.item()) / real.item() }")

            # print(f"Encodings : {encoding.tolist()}")
            data.append(np.array(encoding.tolist()))


    print(np.array(data))

    polyhedra = [(data[0],'tetra'),(data[1],'cube'),(data[2],'oct'),(data[3],'dod'),(data[4],'rotated_tetra'),
                (data[5],'verts_mp567387_Ti_dod'),(data[6],'verts_mp4019_Ti_cube'),(data[7],'verts_mp3397_Ti_tetra'),(data[8],'verts_mp15502_Ba_cube'),(data[9],'verts_mp15502_Ti_oct')]
    distance_similarity_mat = np.zeros(shape = (len(polyhedra),len(polyhedra)))
    cosine_similarity_mat = np.zeros(shape = (len(polyhedra),len(polyhedra)))
    for i,poly_a in enumerate(polyhedra):
        for j,poly_b in enumerate(polyhedra):
            # print('_______________________________________')
            # print(f'Poly_a - {poly_a[1]} | Poly_b - {poly_b[1]}')
            # print(f'Cosine : {cosine_similarity(x=poly_a[0],y=poly_b[0])}')
            # print(f'Distance : {distance_similarity(x=poly_a[0],y=poly_b[0])}')
            distance_similarity_mat[i,j] = distance_similarity(x=poly_a[0],y=poly_b[0]).round(3)
            cosine_similarity_mat[i,j] = cosine_similarity(x=poly_a[0],y=poly_b[0]).round(3)
    print(polyhedra[0][1],polyhedra[1][1],polyhedra[2][1],polyhedra[3][1],polyhedra[4][1],
        polyhedra[5][1],polyhedra[6][1],polyhedra[7][1],polyhedra[8][1],polyhedra[9][1],)
    
    # print(distance_similarty,decimals=3)
    # for x in distance_similarty:
    #     print(x)
    print("--------------------------")
    print("Distance Similarity Matrix")
    print("--------------------------")
    print(distance_similarity_mat)
    print("--------------------------")
    print("Cosine Similarity Matrix")
    print("--------------------------")
    print(cosine_similarity_mat)

    print(count_parameters(model=model))

if __name__ == '__main__':
    main()