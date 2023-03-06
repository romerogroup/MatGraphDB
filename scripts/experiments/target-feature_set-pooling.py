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

from voronoi_statistics.polyhedra_dataset import PolyhedraDataset


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_neurons:List):
        super().__init__()

        layers=[]
        n_layers = len(n_neurons)
        for i_layer in range(n_layers):

            if i_layer == 0:
                layers.append(nn.Linear(n_neurons[0], n_neurons[i_layer]))
            else:
                layers.append(nn.Linear( n_neurons[i_layer] - 1,  n_neurons[i_layer]))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(dropout),)

        layers.append(nn.Linear( n_neurons[-1],  1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PolyhedronModel(nn.Module):

    def __init__(self, n_gc_layers:int=1, n_neurons:List=[2], n_edge_features:int=2, global_pooling_method:str='add'):
        super().__init__()

            
        layers=[]
        for i_gc_layer in range(n_gc_layers):
            if i_gc_layer == 0:
                vals = " x, edge_index, edge_attr -> x0 "
            else:
                vals = " x" + repr(i_gc_layer - 1) + " , edge_index, edge_attr -> x" + repr(i_gc_layer)

            layers.append((pyg_nn.CGConv(n_neurons[0], dim=n_edge_features),vals))

        # self.cg_conv_layers = Sequential(" x, edge_index, edge_attr, batch " , layers)
        self.cg_conv_layers = Sequential(" x, edge_index, edge_attr " , layers)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        # self.ffwd = FeedFoward(n_neurons)
        self.linear_1 = nn.Linear( n_neurons[0],  6)
        self.linear_2 = nn.Linear( 6,  1)

        # self.linear_2 = nn.Linear( 9,  1)

        if global_pooling_method == 'add':
            self.global_pooling_layer = global_add_pool
        elif global_pooling_method == 'mean':
            self.global_pooling_layer = global_mean_pool
        elif global_pooling_method == 'max':
            self.global_pooling_layer = global_max_pool

    def forward(self, x, targets=None):
        
        # Convolutional layers combine nodes and edge interactions
        out = self.cg_conv_layers(x.x, x.edge_index, x.edge_attr ) # out -> (n_total_atoms_in_batch, n_atom_features)
        out = self.sig(out) # out -> (n_total_atoms_in_batch, n_atom_features)

        # # # Fully connected layer. on feature per atom in batch
        out = self.linear_1(out) # out -> (n_total_atoms_in_batch, 1)
        out = self.sig(out)

        
        # batch is index list differteriating which atoms belong to which crystal
        out = self.global_pooling_layer(out, batch = x.batch) # out -> (n_polyhedra, 1)

        out =  self.linear_2(out)
        out = self.relu(out)
        if targets is None:
            loss = None
            mape_loss = None
        else:
            loss_fn = torch.nn.MSELoss()
            mape_loss = mean_absolute_percentage_error(torch.squeeze(out, dim=1), targets)
            loss = loss_fn(torch.squeeze(out, dim=1), targets)

        return out,  loss, mape_loss

    def generate(self, x):
        out = self.cg_conv_layers(x.x, x.edge_index, x.edge_attr )
        out = self.relu(out)
        out = self.linear_1(out) # out -> (n_total_atoms_in_batch, 1)
        out = self.sig(out)
        out = self.global_pooling_layer(out, batch = x.batch)
        
        return out


class EarlyStopping():
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.best_mape_loss = None
        self.counter = 0
        self.status = 0

    def __call__(self, model, val_loss, mape_val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_mape_loss = mape_val_loss
            self.best_model = copy.deepcopy(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.best_mape_loss = mape_val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
        elif self.best_loss - val_loss < self.min_delta:
            self.counter +=1
            if self.counter >= self.patience:
                self.status = f'Stopped on {self.counter}'
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model.state_dict())
                return True
            self.status = f"{self.counter}/{self.patience}"
            # print( self.status )
        return False

def main():
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    top_dir = os.path.dirname(experiment_dir)

    # feasture_set_index = 0
    dataset = 'material_random_polyhedra'


    # Training params
    train_model = True
    n_epochs = 1000
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Other params
    batch_size = 2
    dropout = 0.2


    y_vals = ['dihedral_energy', 'moi', 'energy_per_verts']
    feature_set_indices = [0,1,2,3,4,5]
    global_pooling_methods = ['mean','add','max']
    
    for feasture_set_index in feature_set_indices:
        train_dir = f"{top_dir}{os.sep}datasets{os.sep}{dataset}{os.sep}feature_set_{feasture_set_index}{os.sep}train"
        test_dir = f"{top_dir}{os.sep}datasets{os.sep}{dataset}{os.sep}feature_set_{feasture_set_index}{os.sep}test"
        val_dir = f"{top_dir}{os.sep}datasets{os.sep}{dataset}{os.sep}feature_set_{feasture_set_index}{os.sep}val"

        val_dataset = PolyhedraDataset(database_dir=val_dir,device=device, y_val='moi')
        n_node_features = val_dataset[0].x.shape[1]
        n_edge_features = val_dataset[0].edge_attr.shape[1]
        del val_dataset

        experiment_name = f'target value_feature_set_{feasture_set_index}_{dataset}_experiment'
        mlflow_dir = f"file:{top_dir}{os.sep}mlruns"
        mlflow.set_tracking_uri(mlflow_dir)
        
        # deleting experiment if it exsits
        experiments_list = mlflow.search_experiments()
        for experiment in experiments_list:
            if experiment.name == experiment_name:
                mlflow.delete_experiment(experiment.experiment_id)
        experiment_id = mlflow.create_experiment(experiment_name)

        for global_pooling_method in global_pooling_methods:
            for y_val in y_vals:
                # Loading directories as Datatsets
                train_dataset = PolyhedraDataset(database_dir=train_dir,device=device, y_val=y_val)
                val_dataset = PolyhedraDataset(database_dir=val_dir,device=device, y_val=y_val)
                test_dataset = PolyhedraDataset(database_dir=test_dir,device=device, y_val=y_val)

                n_train = len(train_dataset)
                n_validation = len(val_dataset)

                # Creating data loaders
                train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

                run_name = f'{y_val}_pooling-{global_pooling_method}'
                with mlflow.start_run(experiment_id=experiment_id,run_name=run_name):
                    model = PolyhedronModel(n_gc_layers = 1, n_neurons=[n_node_features], n_edge_features=n_edge_features,global_pooling_method=global_pooling_method)
                    m = model.to(device)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                    es = EarlyStopping(patience = 10)

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
                        for i, sample in enumerate(val_loader):
                            torch.set_grad_enabled(False)
                            out, val_loss, mape_val_loss = model(sample , targets = sample.y)
                            torch.set_grad_enabled(True)
                            batch_val_loss += val_loss.item()
                            batch_val_mape += mape_val_loss.item()
                        batch_val_loss = batch_val_loss / (i+1)
                        batch_val_mape = batch_val_mape / (i+1)


                        batch_test_loss = 0.0
                        batch_test_mape = 0.0
                        for i, sample in enumerate(test_loader):
                            torch.set_grad_enabled(False)
                            out, test_loss, mape_test_loss = model(sample , targets = sample.y)
                            torch.set_grad_enabled(True)
                            batch_test_loss += test_loss.item()
                            batch_test_mape += mape_test_loss.item()
                        batch_test_loss = batch_test_loss / (i+1)
                        batch_test_mape = batch_test_mape / (i+1)
                        # val_loss *= (factor)  # to put it on the same scale as the training running loss)
                        # print(repr(n_epoch) + ",  " + repr(batch_train_loss) + ",  " + repr(batch_val_loss))

                        mlflow.log_metric('mse_loss',batch_train_loss,step=1)
                        mlflow.log_metric('mse_val_loss',batch_val_loss,step=1)
                        mlflow.log_metric('mse_val_loss',batch_test_loss,step=1)

                        mlflow.log_metric('mae_loss',batch_train_loss**0.5,step=1)
                        mlflow.log_metric('mae_val_loss',batch_val_loss**0.5,step=1)
                        mlflow.log_metric('mae_test_loss',batch_test_loss**0.5,step=1)

                        mlflow.log_metric('mape_loss',batch_train_mape,step=1)
                        mlflow.log_metric('mape_val_loss',batch_val_mape,step=1)
                        mlflow.log_metric('mape_test_loss',batch_test_mape,step=1)
                        

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

                


                # mlflow.pytorch.log_model(model,'model')

    return None
if __name__ == '__main__':
    main()

