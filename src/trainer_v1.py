import os
import copy
import json
import yaml
from datetime import datetime
import pickle
import time
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
from torch.nn import functional as F
from torch_geometric.nn import CGConv, global_add_pool, global_mean_pool, global_max_pool, Sequential
from torch_geometric.loader import DataLoader,DataListLoader
import torch_geometric.nn as pyg_nn

from torch.nn import functional as F
import torch.multiprocessing as mp

from residual_graph_potential import PolyhedronResidualModel
from features import Features, set_up_features
from set_up_atomic_structure_graphs import set_up_atomic_structure_graphs
from graph_dataset import GraphDataSet
from callbacks import EarlyStopping
# from QM9_covalent_molecular_graphs import QM9CovalentMolecularGraphs

large_width = 400
np.set_printoptions(linewidth=large_width)

###################################################################
# Parameters
###################################################################
def main():

    # mp.set_start_method('spawn')

    torch.manual_seed(0)
    project_dir = os.path.dirname(__file__)
    print(project_dir)
    input_file = os.path.join(project_dir,'sample_residual.yml')
    with open( input_file, 'r' ) as input_stream:
        SETTINGS = yaml.load(input_stream, Loader=yaml.Loader)

    edges, bond_angle, dihedral_angle = set_up_features(input_data = SETTINGS)

    edge_parameters = edges.parameters()
    bond_angle_parameters = bond_angle.parameters()

    if dihedral_angle:
        dihedral_angle_parameters = dihedral_angle.parameters()

    # ###################################################################
    # # Start of the the training run
    # ###################################################################


    # Loading data -------------------------------------------------------------------------------

    graphs = set_up_atomic_structure_graphs(
        SETTINGS['graph_type'],
        species=SETTINGS['species'],
        edge_features=edges,
        bond_angle_features=bond_angle,
        dihedral_features=dihedral_angle,
        node_feature_list=SETTINGS['node_features'],
        n_total_node_features=SETTINGS['n_total_node_features'],
    )

    trainDataset = GraphDataSet(
        SETTINGS['trainDir'], 
        graphs, 
        nMaxEntries=SETTINGS['n_train_max_entries'], 
        seed=SETTINGS['seed'], 
        transform=SETTINGS['transform'],
        file_extension = SETTINGS['file_extension']
    )

    if SETTINGS['n_train_max_entries']:
        n_train = SETTINGS['n_train_max_entries']
    else:
        n_train = len(trainDataset)

    valDataset = GraphDataSet(
        SETTINGS['valDir'], 
        graphs, 
        nMaxEntries=SETTINGS['n_val_max_entries'], 
        seed=SETTINGS['seed'], 
        transform=SETTINGS['transform'],
        file_extension = SETTINGS['file_extension']
    )

    if SETTINGS['n_val_max_entries']:
        n_val = SETTINGS['n_val_max_entries']
    else:
        n_val = len(valDataset)

    train_loader = DataLoader(trainDataset, batch_size=SETTINGS['batch_size'], num_workers=SETTINGS['num_workers'])
    val_loader = DataLoader(valDataset, batch_size=SETTINGS['batch_size'], num_workers=SETTINGS['num_workers'])

    n_node_features = valDataset[0].x.shape[1]
    n_edge_features = valDataset[0].edge_attr.shape[1]
    avg_y_val_train = None

    print("Number of gpus available : ", torch.cuda.device_count())
    print("Using device : ", torch.cuda.is_available())
    print("Number of Validation samples: " + str(len(trainDataset)))
    print("Number of Validation samples: " + str(len(valDataset)))
    print("Number of node features : ", n_node_features)
    print("Number of edge features : ", n_edge_features)

    # Initializing model and callbacks -------------------------------------------------------------------------------


    model = PolyhedronResidualModel(n_node_features=n_node_features, 
                                    n_edge_features=n_edge_features, 
                                    n_gc_layers=SETTINGS['n_gc_layers'],
                                    layers_1=SETTINGS['layers_1'],
                                    layers_2=SETTINGS['layers_2'],
                                    dropout=SETTINGS['dropout'],
                                    apply_layer_norms=SETTINGS['apply_layer_norms'],
                                    global_pooling_method=SETTINGS['global_pooling_method'],
                                    target_mean=avg_y_val_train)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Number of trainable parameters: " + str(pytorch_total_params))
    print("Number of training samples: " + str(n_train) )
    print("Number of val samples: " + str(n_val) )

    m = model.to(SETTINGS['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(SETTINGS['learning_rate']))

    es = EarlyStopping(patience = SETTINGS['callbacks']['earlyStopping']['patience'])

    loss_fn = torch.nn.MSELoss()
    # Defining Training loop -------------------------------------------------------------------------------


    ###################################################################
# Train and Test epoch steps
###################################################################

    def train(single_batch:bool=False):
        batch_train_loss = 0.0
        for i,sample in enumerate(train_loader):
            sample.to(SETTINGS['device'])
            if single_batch:
                if i == 0:
                    optimizer.zero_grad()
                    targets = sample.y
                    out = model(sample)
                    train_loss = loss_fn(torch.squeeze(out, dim=1), targets)
                    train_loss.backward()
                    optimizer.step()
                    batch_train_loss += train_loss.item()

            else:
                optimizer.zero_grad()
                targets = sample.y
                out = model(sample)
                train_loss = loss_fn(torch.squeeze(out, dim=1), targets)
                train_loss.backward()
                optimizer.step()
                batch_train_loss += train_loss.item()

        batch_train_loss = batch_train_loss / (i+1)
        return batch_train_loss

    @torch.no_grad()
    def test():
        model.eval()
        batch_test_loss = 0.0
        for i,sample in enumerate(test_loader):
            sample.to(SETTINGS['device'])
            targets = sample.y
            out = model(sample)
            test_loss = loss_fn(torch.squeeze(out, dim=1), targets)
            batch_test_loss += test_loss.item()
        batch_test_loss = batch_test_loss / (i+1)
        model.train()
        return batch_test_loss
    



    metrics_dict = {
                "epochs":np.arange(SETTINGS['epochs']),
                "train_mse":[],
                "val_mse":[],
                "trainable_params":pytorch_total_params,
                }

    n_epoch_0 = 0
    model.train()
    start_time = time.time()
    for epoch in range(SETTINGS['epochs']):
        n_epoch = n_epoch_0 + epoch
        batch_train_loss = train(single_batch=SETTINGS['single_batch'])
        batch_test_loss = test()

        metrics_dict['train_mse'].append(batch_train_loss)
        metrics_dict['val_mse'].append(batch_test_loss)


        if es is not None:
            if es(model=model, val_loss=batch_test_loss):
                print("Early stopping")
                print('_______________________')
                print(f'Stopping : {epoch - es.counter}')
                print(f'mae_val : {es.best_loss**0.5}')
                break

        if n_epoch % 1 == 0:
            print(f"{n_epoch}, {batch_train_loss:.5f}, {batch_test_loss:.5f}")



    # Saving model metrics and parameters

    os.makedirs(SETTINGS['save_dir'], exist_ok=True)
    for n in range(1, 9999):
        p = SETTINGS['save_dir'] + os.sep + f'train{n}'  # increment path
        if not os.path.exists(p):  #
            break

    run_dir = p
    weights_dir = run_dir + os.sep + 'weights'
    os.makedirs(weights_dir)

    args_file = os.path.join(run_dir,'sample_residual.yml')
    with open( args_file, 'w' ) as file:
        yaml.dump(SETTINGS, file)

    txt_file = os.path.join(run_dir,'training.txt')
    training_time = time.time() - start_time
    with open(txt_file,'w') as file:
        file.write(f"Training time : {training_time:.2f}s")

    metrics = pd.DataFrame(metrics_dict)

    metrics_file = os.path.join(run_dir,'results.csv')
    metrics.to_csv(metrics_file, index=False)



    ckpt_last = {
                'epoch': SETTINGS['epochs'],
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_args': SETTINGS,  # save as dict
                'date': datetime.now().isoformat()}

    torch.save(ckpt_last, weights_dir +os.sep+ f'last.pt', pickle_module=pickle)

    ckpt_last = {
                'epoch': SETTINGS['epochs'],
                'model': es.best_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_args': SETTINGS,  # save as dict
                'date': datetime.now().isoformat()}

    torch.save(ckpt_last, weights_dir +os.sep+ f'best.pt', pickle_module=pickle)

if __name__ == '__main__':
    main()

    