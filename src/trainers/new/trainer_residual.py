import os
import copy
from typing import List
import json
import yaml
import pickle
from datetime import datetime
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
from torch.nn import functional as F
from torch_geometric.nn import CGConv, global_add_pool, global_mean_pool, global_max_pool, Sequential
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn
from torchmetrics.functional import mean_absolute_percentage_error
from torch.nn import functional as F

from poly_graphs_lib.data.dataset import MPPolyDataset
from poly_graphs_lib.callbacks import EarlyStopping
from poly_graphs_lib.models.poly.poly_residual_regression_model import PolyhedronResidualModel
from poly_graphs_lib.utils.plotting import plot_training_curves
from poly_graphs_lib.utils.math_dataset import target_statistics
from poly_graphs_lib.utils.timing import Timer
from poly_graphs_lib.utils.torch_utils import get_total_dataset_bytes,get_model_bytes
from poly_graphs_lib.utils.metrics import compare_polyhedra
from poly_graphs_lib.utils import ROOT,PROJECT_DIR


np.set_printoptions(linewidth=40)

class Trainer:
    def __init__(self):
        pass

    def load_config(self, input_file):
        print('-------------------------------------------------------------------------')
        print('===============================')
        print("Initializing configurations")
        print('===============================')

        print("Config file : ", input_file)
        with open( input_file, 'r' ) as input_stream:
            self.settings = yaml.load(input_stream, Loader=yaml.Loader)

        torch.manual_seed(self.settings['seed'])

    def initialize_datasets(self):
        print('-------------------------------------------------------------------------')
        print('===============================')
        print("Initializing Datasets")
        print('===============================')
        if self.settings is None:
            raise ("Call load_config first")


        self.train_dataset = MPPolyDataset(save_root=self.settings['train_dir'])
        self.test_dataset = MPPolyDataset(save_root=self.settings['test_dir'])
        
        if self.settings['n_train_max_entries']:
            print(self.settings['n_train_max_entries'])
            self.train_dataset = self.train_dataset[:self.settings['n_train_max_entries']]

        self.n_node_features=self.train_dataset[0].x.shape[1]
        self.n_edge_features=self.train_dataset[0].edge_attr.shape[1]

        self.n_train = len(self.train_dataset)
        self.n_test = len(self.test_dataset)

        print("Using cuda : ", torch.cuda.is_available())
        print("Number of gpus available : ", torch.cuda.device_count())
        print("Number of Training samples: " + str(self.n_train))
        print("Number of Test samples: " + str(self.n_test))
        print("Number of node features : ", self.n_node_features)
        print("Number of edge features : ", self.n_edge_features)

        if self.settings['calculate_target_stats']:
            self.calculate_dataset_stats()

        # Creating data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.settings['batch_size'],shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.settings['batch_size'],shuffle=False)

    def calculate_dataset_stats(self):
        print('________________________')
        print('Target values statistics')
        print('________________________')
        if self.train_dataset is None:
            raise ("Call initialize_datasets first")
        self.avg_y_train, self.std_y_train = target_statistics(self.train_dataset,device=self.settings['device'])
        self.avg_y_test, self.std_y_test = target_statistics(self.test_dataset,device=self.settings['device'])

        print(f"Train average y_val: {self.avg_y_train}")
        print(f"Train std y_val: {self.std_y_train}")
        print(f"Test average y_val: {self.avg_y_test}")
        print(f"Test std y_val: {self.std_y_test}")

    def initialize_model(self):
        print('-------------------------------------------------------------------------')
        print('===============================')
        print("Initializing Model")
        print('===============================')

        if self.n_node_features is None:
            raise ("Call initialize datasets first")
        
        self.model = PolyhedronResidualModel(n_node_features=self.n_node_features, 
                                    n_edge_features=self.n_edge_features, 
                                    n_gc_layers=self.settings['n_gc_layers'],
                                    layers_1=self.settings['layers_1'],
                                    layers_2=self.settings['layers_2'],
                                    dropout=self.settings['dropout'],
                                    apply_layer_norms=self.settings['apply_layer_norms'],
                                    global_pooling_method=self.settings['global_pooling_method'],
                                    target_mean=None)
        

        # self.model=self.model.double()
        self.model.to(self.settings['device'])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.settings['learning_rate']))
        self.es = EarlyStopping(patience = self.settings['early_stopping_patience'])
        self.loss_fn = torch.nn.MSELoss()

        self.metrics_dict = {
                "epochs":'',
                "train_mse":[],
                "test_mse":[],
                }

        print(self.model)
        print("Optimizer : " + str(self.optimizer))
        print("Loss function : " + str(self.loss_fn))
        print("Early Stoppying Patience : " + str(self.settings['early_stopping_patience']))
        
        print('_____________________')
        print("Metrics")
        print('_____________________')
        for key,value in self.metrics_dict.items():
            print(f"{key}")

    def get_required_memory(self):
        print('-------------------------------------------------------------------------')
        print('===============================')
        print("Profiling memory usage")
        print('===============================')
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.total_bytes = 0

        self.train_bytes = get_total_dataset_bytes(self.train_dataset)
        self.test_bytes = get_total_dataset_bytes(self.test_dataset)
        self.model_bytes=get_model_bytes(self.model)

        self.total_bytes+=self.train_bytes
        self.total_bytes+=self.test_bytes
        self.total_bytes+=self.model_bytes

        print(f"Train memory : {self.train_bytes/1024**3} GB")
        print(f"Test memory : {self.test_bytes/1024**3} GB")
        print(f"Model memory : {self.model_bytes/1024**3} GB")

        print(f"Total memory : {self.total_bytes/1024**3} GB")

    def train_loop(self):
        self.model.train()
        batch_train_loss = 0.0
        for i,sample in enumerate(self.train_loader):
            sample.to(self.settings['device'])
            if self.settings['single_batch']:
                if i == 0:
                    self.optimizer.zero_grad()
                    targets = sample.y
                    out = self.model(sample)

                    train_loss = self.loss_fn(torch.squeeze(out, dim=1), targets)
                    train_loss.backward()
                    self.optimizer.step()
                    batch_train_loss += train_loss.item()

            else:
                self.optimizer.zero_grad()
                targets = sample.y
                out = self.model(sample)
                
                train_loss = self.loss_fn(torch.squeeze(out, dim=1), targets)
                train_loss.backward()
                self.optimizer.step()
                batch_train_loss += train_loss.item()

        batch_train_loss = batch_train_loss / (i+1)
        return batch_train_loss
    
    @torch.no_grad()
    def test_loop(self):
        self.model.eval()
        batch_test_loss = 0.0
        for i,sample in enumerate(self.test_loader):
            sample.to(self.settings['device'])
            targets = sample.y
            out = self.model(sample)
            test_loss = self.loss_fn(torch.squeeze(out, dim=1), targets)
            batch_test_loss += test_loss.item()
        batch_test_loss = batch_test_loss / (i+1)
        self.model.train()
        return batch_test_loss
    
    def train(self):
        print('-------------------------------------------------------------------------')
        print('===============================')
        print("Starting training loop")
        print('===============================')
        n_epoch_0 = 0
        self.model.train()
        for epoch in range(self.settings['epochs']):
            n_epoch = n_epoch_0 + epoch

            batch_train_loss = self.train_loop()
            batch_test_loss= self.test_loop()

            self.metrics_dict['train_mse'].append(batch_train_loss)
            self.metrics_dict['test_mse'].append(batch_test_loss)

            if self.es is not None:
                if self.es(model=self.model, val_loss=batch_test_loss):
                    print("Early stopping")
                    print('_______________________')
                    print(f'Stopping : {epoch - self.es.counter}')
                    print(f'mae_val : {self.es.best_loss**0.5}')
                    break

            if n_epoch % 1 == 0:
                print(f"{n_epoch}, {batch_train_loss:.5f}, {batch_test_loss:.5f}")

        self.metrics_dict['epochs'] = np.arange(len(self.metrics_dict['train_mse'])) 
        print('===============================')
        print("Ending training loop")
        print('===============================')

    def save_training(self):
        print('-------------------------------------------------------------------------')
        print('===============================')
        print("Saving results in the run directory")
        print('===============================')

        # Creating training save dir
        os.makedirs(self.settings['save_dir'], exist_ok=True)
        for n in range(1, 9999):
            p = self.settings['save_dir'] + os.sep + f'train{n}'  # increment path
            if not os.path.exists(p):  #
                break
        
        self.run_dir = p
        weights_dir = self.run_dir + os.sep + 'weights'
        os.makedirs(weights_dir)

        # plotting training curves
        plot_training_curves(epochs=self.metrics_dict['epochs'], 
                            train_loss=self.metrics_dict['train_mse'], 
                            val_loss=None, 
                            test_loss=self.metrics_dict['test_mse'],
                            loss_label='MSE', 
                            filename=f'{self.run_dir}{os.sep}training_curve.png',
                            log_scale=True)

        # plotting polyhedra comparison
        compare_polyhedra(run_dir = self.run_dir,dataset=self.test_dataset, model=self.model,device=self.settings['device'])

        args_file = os.path.join(self.run_dir,'sample.yml')

        # Saving settings
        with open( args_file, 'w' ) as file:
            yaml.dump(self.settings, file)

        # Save metrics to csv
        metrics = pd.DataFrame(self.metrics_dict)
        metrics_file = os.path.join(self.run_dir,'results.csv')
        metrics.to_csv(metrics_file, index=False)


        # Save best and last model checkpoints
        ckpt_last = {
                    'epoch': self.settings['epochs'],
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'train_args': self.settings,  # save as dict
                    'date': datetime.now().isoformat()}

        torch.save(ckpt_last, weights_dir +os.sep+ f'last.pt', pickle_module=pickle)

        ckpt_last = {
                    'epoch': self.settings['epochs'],
                    'model': self.es.best_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'train_args': self.settings,  # save as dict
                    'date': datetime.now().isoformat()}

        torch.save(ckpt_last, weights_dir+os.sep+ f'best.pt', pickle_module=pickle)

        # Saving memory estimate
        if self.total_bytes:
            with open(os.path.join(self.run_dir,'memeory.txt') , 'w') as f:
                f.write(f"Training Dataset memory : {self.train_bytes/1024**3} GB\n")
                f.write(f"Test Dataset memory : {self.test_bytes/1024**3} GB\n")
                f.write(f"Model memory : {self.model_bytes/1024**3} GB\n")
                f.write("___________________________________________\n")
                f.write(f"Total memory : {self.total_bytes/1024**3} GB\n")

        # Saving target statistics
        if self.avg_y_train:
            with open(os.path.join(self.run_dir,'target_statistics.txt') , 'w') as f:
                f.write(f"Train average y_val: {self.avg_y_train}\n")
                f.write(f"Train std y_val: {self.std_y_train}\n")
                f.write(f"Test average y_val: {self.avg_y_test}\n")
                f.write(f"Test std y_val: {self.std_y_test}\n")

        print(f"Saving run to : {self.run_dir}")



def main():
    print(PROJECT_DIR)
    config_file = os.path.join(PROJECT_DIR,'src','trainers','new','residual_dataset_1.yml')

    # config_file = os.path.join(PROJECT_DIR,'src','trainers','new','residual_dataset_2.yml') 

    
    timer = Timer()
    trainer=Trainer()

    timer.start_event("Initializing Configurations")
    trainer.load_config(input_file=config_file)
    timer.end_event()

    timer.start_event("Initializing Datasets")
    trainer.initialize_datasets()
    timer.end_event()

    timer.start_event("Initializing Model")
    trainer.initialize_model()
    timer.end_event()
    
    timer.start_event("Estimating Required Memory")
    trainer.get_required_memory()
    timer.end_event()

    timer.start_event("Starting Train Loop")
    trainer.train()
    timer.end_event()

    timer.start_event("Saving in Train directory")
    trainer.save_training()
    timer.end_event()

    timer.save_events(filename=os.path.join(trainer.run_dir,"time_profile.txt"))



if __name__ == '__main__':
    main()