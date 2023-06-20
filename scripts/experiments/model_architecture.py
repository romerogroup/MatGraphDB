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


from poly_graphs_lib.pyg_dataset import PolyhedraDataset
from poly_graphs_lib.callbacks import EarlyStopping
from poly_graphs_lib.models.poly_regression_model import PolyhedronModel

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
        self.linear_2 = nn.Linear( n_hidden_layers[0], n_hidden_layers[1])
        self.out_layer= nn.Linear( n_hidden_layers[-1],  1)
        
        # self.layer_norm = nn.LayerNorm(self.angle_fea_len)

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

        x_out = self.bn_node(x)
        edge_out = self.bn_edge(edge_attr)
 
        # Convolutional layers combine nodes and edge interactions
        out = self.cg_conv_layers(x_out, edge_index, edge_out ) # out -> (n_total_node_in_batch, n_node_features)
        out = self.leaky_relu(out)

        # Fully connected layer
        out = self.linear_1(out) # out -> (n_total_nodes_in_batch, n_hidden_layers[0])
        out = self.leaky_relu(out) # out -> (n_total_nodes_in_batch, n_hidden_layers[0])

        # batch is index list differteriating which nodes belong to which graph
        out = self.global_pooling_layer(out, batch = batch) # out -> (n_graphs, n_hidden_layers[0])
        
        out = self.linear_2(out) 
        out = self.leaky_relu(out) 

        out = self.out_layer(out) # out -> (n_graphs, 1)
        out = self.relu(out) # out -> (n_graphs, 1)

        # Loss handling
        if targets is None:
            loss = None
            mape_loss = None
        else:
            loss_fn = torch.nn.MSELoss()
            # loss_fn = torch.nn.MAELoss()
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
        out = self.global_pooling_layer(out, batch = x.batch)
        out = self.linear_2(out)
        out = self.leaky_relu(out) 
        return out
    

def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # hyperparameters
    save_model = True

    # Training params
    n_epochs = 1000
    learning_rate = 1e-2
    batch_size = 8
    early_stopping_patience = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # polyhedron model parameters
    n_gc_layers = 3
    n_hidden_layers=[4,3]
    global_pooling_method = 'add'

    # dataset parameters
    dataset = 'material_random_polyhedra'
    feasture_set_index = 3
    y_val = ['energy_per_verts','dihedral_energy'][0]
    
    ###################################################################
    # Start of the the training run
    ###################################################################

    train_dir = f"{project_dir}{os.sep}datasets{os.sep}{dataset}{os.sep}feature_set_{feasture_set_index}{os.sep}train"
    test_dir = f"{project_dir}{os.sep}datasets{os.sep}{dataset}{os.sep}feature_set_{feasture_set_index}{os.sep}test"
    val_dir = f"{project_dir}{os.sep}datasets{os.sep}{dataset}{os.sep}feature_set_{feasture_set_index}{os.sep}val"

    val_dataset = PolyhedraDataset(database_dir=val_dir, device=device, y_val=y_val)
    n_node_features = val_dataset[0].x.shape[1]
    n_edge_features = val_dataset[0].edge_attr.shape[1]
    del val_dataset

    experiment_name = f'model_architecture'
    architecture_num=0
    mlflow_dir = f"file:{project_dir}{os.sep}mlruns"
    mlflow.set_tracking_uri(mlflow_dir)
    
    # deleting experiment if it exsits
    experiments_list = mlflow.search_experiments()
    for experiment in experiments_list:
        if experiment.name == experiment_name:
            mlflow.delete_experiment(experiment.experiment_id)
    experiment_id = mlflow.create_experiment(experiment_name)

    
    train_dataset = PolyhedraDataset(database_dir=train_dir,device=device, y_val=y_val)
    # train_2_dataset = PolyhedraDataset(database_dir=train_2_dir,device=device, y_val=y_val)
    test_dataset = PolyhedraDataset(database_dir=test_dir,device=device, y_val=y_val)
    val_dataset = PolyhedraDataset(database_dir=val_dir,device=device, y_val=y_val)

    n_train = len(train_dataset)
    n_validation = len(val_dataset)

    # Creating data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    # train_2_loader = DataLoader(train_2_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    run_name = f'architecture-{architecture_num}'
    with mlflow.start_run(experiment_id=experiment_id,run_name=run_name):
        model = PolyhedronModel(n_node_features=n_node_features, 
                                n_edge_features=n_edge_features, 
                                n_gc_layers=n_gc_layers,
                                n_hidden_layers=n_hidden_layers,
                                global_pooling_method=global_pooling_method)
        m = model.to(device)
        print(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        es = EarlyStopping(patience = early_stopping_patience)

        target_values = []
        for sample in train_loader:
            target_values.extend(sample.y.tolist())
        target_values= torch.tensor(target_values,device = device)

        mlflow.log_param('min_y_val',float(target_values.min().item()))
        mlflow.log_param('max_y_val',float(target_values.max().item()))
        mlflow.log_param('mean_y_val',float(target_values.mean().item()))
        mlflow.log_param('std_y_val',float(target_values.std().item()))

        n_epoch_0 = 0
        model.train()
        for epoch in range(n_epochs):
            n_epoch = n_epoch_0 + epoch
            batch_train_loss = 0.0
            batch_train_mape = 0.0
            for i,sample in enumerate(train_loader):
                optimizer.zero_grad()
                out, train_loss, mape_loss = model(sample , targets = sample.y)
                train_loss.backward()
                optimizer.step()
                batch_train_loss += train_loss.item()
                batch_train_mape += mape_loss.item()

            batch_train_loss = batch_train_loss / (i+1)
            batch_train_mape = batch_train_mape / (i+1)

            batch_val_loss = 0.0
            batch_val_mape = 0.0
            for i,sample in enumerate(val_loader):
                torch.set_grad_enabled(False)
                out, val_loss, mape_val_loss = model(sample , targets = sample.y)
                torch.set_grad_enabled(True)
                batch_val_loss += val_loss.item()
                batch_val_mape += mape_val_loss.item()
            batch_val_loss = batch_val_loss / (i+1)
            batch_val_mape = batch_val_mape / (i+1)


            batch_test_loss = 0.0
            batch_test_mape = 0.0
            for i,sample in enumerate(test_loader):
                torch.set_grad_enabled(False)
                out, test_loss, mape_test_loss = model(sample , targets = sample.y)
                torch.set_grad_enabled(True)
                batch_test_loss += test_loss.item()
                batch_test_mape += mape_test_loss.item()
            batch_test_loss = batch_test_loss / (i+1)
            batch_test_mape = batch_test_mape / (i+1)


            # val_loss *= (factor)  # to put it on the same scale as the training running loss)
            if n_epoch % 10 == 0:
                print(repr(n_epoch) + ",  " + repr(batch_train_loss) + ",  " + repr(batch_val_loss)+ ",  " + repr(batch_test_loss))

            mlflow.log_metric('mse_loss',batch_train_loss,step=1)
            mlflow.log_metric('mse_val_loss',batch_val_loss,step=1)

            mlflow.log_metric('mae_loss',batch_train_loss**0.5,step=1)
            mlflow.log_metric('mae_val_loss',batch_val_loss**0.5,step=1)

            mlflow.log_metric('mape_loss',batch_train_mape,step=1)
            mlflow.log_metric('mape_val_loss',batch_val_mape,step=1)
            

            if es(model=model, val_loss=batch_val_loss,mape_val_loss=batch_val_mape):
                # mlflow.log_metric('best_mse_loss',batch_train_loss)
                mlflow.log_metric('best_mse_val_loss',es.best_loss)
                # mlflow.log_metric('best_mae_loss',batch_train_loss**0.5)
                mlflow.log_metric('best_mae_val_loss',es.best_loss**0.5)
                # mlflow.log_metric('best_mape_loss',batch_val_mape)
                mlflow.log_metric('best_mape_val_loss',es.best_mape_loss)
                mlflow.log_metric('stopping_epoch',epoch - es.counter)
                break



        mlflow.log_param('target_value',y_val)
        mlflow.log_param('learning_rate',learning_rate)
        mlflow.log_param('n_epochs',n_epochs)
        
        mlflow.log_param(f'feature_set',feasture_set_index)
        mlflow.log_param('global_pooling_method',global_pooling_method)

        if save_model:
            checkpoint = {
                    "epoch":n_epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    }
            torch.save(checkpoint, f"{project_dir}{os.sep}models{os.sep}model_checkpoint_1000.pth")

    return None
if __name__ == '__main__':
    main()

