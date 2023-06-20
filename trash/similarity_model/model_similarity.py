import os
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import CGConv, global_add_pool, global_mean_pool, Sequential
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn

from polyhedra_dataset import PolyhedraDataset


# hyperparameters
# directories where data is
parent_dir = os.path.dirname(__file__)
train_dir = f"{parent_dir}{os.sep}train"
test_dir = f"{parent_dir}{os.sep}test"
val_dir = f"{parent_dir}{os.sep}val"



# Saving and Loading model settings
model_name = 'model_checkpoint_three_body_energy_1.pth'
# ----------------------
# Saving from scratch
load_checkpoint = None
save_model = f"{parent_dir}{os.sep}{model_name}"
# ----------------------
# Loading model
# load_checkpoint = f"{parent_dir}{os.sep}{model_name}"
# save_model = None

# Training params
train_model = True
n_epochs = 1000
learning_rate = 1e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Other params
batch_size = 16
dropout = 0.2

n_node_features = 2
n_edge_features = 2
# ------------

# torch.manual_seed(1337)
torch.manual_seed(1330)

# Loading directories as Datatsets
train_dataset = PolyhedraDataset(database_dir=train_dir,device=device)
val_dataset = PolyhedraDataset(database_dir=val_dir,device=device)

n_train = len(train_dataset)
n_validation = len(val_dataset)

# Creating data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

# for sample in train_loader:
#     # print(sample)
#     for data in sample:
#         print(data)


# class FeedFoward(nn.Module):
#     """ a simple linear layer followed by a non-linearity """

#     def __init__(self, n_neurons:List):
#         super().__init__()

#         layers=[]
#         n_layers = len(n_neurons)
#         for i_layer in range(n_layers):

#             if i_layer == 0:
#                 layers.append(nn.Linear(n_node_features, n_neurons[i_layer]))
#             else:
#                 layers.append(nn.Linear( n_neurons[i_layer] - 1,  n_neurons[i_layer]))
#             layers.append(nn.ReLU())
#             # layers.append(nn.Dropout(dropout),)

#         layers.append(nn.Linear( n_neurons[-1],  1))
        
#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
        # return self.net(x)

class PolyhedronModel(nn.Module):

    def __init__(self, n_gc_layers:int=1, n_neurons:List=[n_node_features]):
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
        self.l1 = nn.Linear(n_node_features,3)
        self.l2 = nn.Linear(3*2, 1)
        # self.ffwd = FeedFoward(n_neurons)
        self.global_pooling_layer = global_mean_pool

    def forward(self, x, targets=None):
        # print(x[0])
        # print(x.shape)
        # for val in x:
        

        poly_a, poly_b, y = x

        # print(poly_a)
        # Convolutional layers combine nodes and edge interactions
        out_1 = self.cg_conv_layers(poly_a.x, poly_a.edge_index, poly_a.edge_attr ) # out -> (n_total_atoms_in_batch, n_atom_features)
        # print(out_1.shape)
        out_2 = self.cg_conv_layers(poly_b.x, poly_b.edge_index, poly_b.edge_attr ) # out -> (n_total_atoms_in_batch, n_atom_features)
        # print(out_2.shape)

        out_1 = self.l1(out_1) # out -> (n_total_atoms_in_batch, 10)
        out_1 = self.relu(out_1) # out -> (n_total_atoms_in_batch, 10)
        # batch is index list differteriating which atoms belong to which crystal
        out_1 = self.global_pooling_layer(out_1, batch = poly_a.batch) # out -> (n_polyhedra, 10)

        out_2 = self.l1(out_2) # out -> (n_total_atoms_in_batch, 10)
        out_2 = self.relu(out_2) # out -> (n_total_atoms_in_batch, 10)
        # batch is index list differteriating which atoms belong to which crystal
        out_2 = self.global_pooling_layer(out_2, batch = poly_b.batch) # out -> (n_polyhedra, 10)

        # print(out_1.shape,out_2.shape)

        # out = out_1+out_2
        out = torch.cat((out_1, out_2), 1)

        # print(out.shape)    
        out = self.l2(out) # out -> (n_polyhedra, 1)
        # print(out)
        out = self.relu(out)
        if targets is None:
            loss = None
        else:
            loss_fn = torch.nn.MSELoss()
            # print(targets)
            # print(torch.squeeze(out, dim=1))
            loss = loss_fn(torch.squeeze(out, dim=1), targets)
            # print(loss)

        return out,  loss

    def generate(self, x):
        out = self.cg_conv_layers(x.x, x.edge_index, x.edge_attr )
        out = self.l1(out) # out -> (n_total_atoms_in_batch, 10)
        out = self.relu(out) # out -> (n_total_atoms_in_batch, 10)
        out = self.global_pooling_layer(out, batch = x.batch) # out -> (n_polyhedra, 10)
        return out

    def predict(self, x):
        out = self.cg_conv_layers(x.x, x.edge_index, x.edge_attr )
        out = self.l1(out) # out -> (n_total_atoms_in_batch, 10)
        out = self.relu(out) # out -> (n_total_atoms_in_batch, 10)
        out = self.global_pooling_layer(out, batch = x.batch) # out -> (n_polyhedra, 10)
        out = self.l2(out) # out -> (n_polyhedra, 1)
        out = self.relu(out)
        return out



def main():

    # Either loading a model or starting from scratch
    if load_checkpoint:
        loaded_checkpoint = torch.load(load_checkpoint)
        
        model = PolyhedronModel(n_gc_layers = 1, n_neurons=[n_node_features])
        m = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        model.load_state_dict(loaded_checkpoint['model_state'])
        optimizer.load_state_dict(loaded_checkpoint['optim_state'])
        n_epoch = loaded_checkpoint['epoch']
        train_loss = loaded_checkpoint['train_loss']
        val_loss = loaded_checkpoint['val_loss']
        print(repr(n_epoch) + ",  " + repr(train_loss) + ",  " + repr(val_loss))
        for sample in train_loader:
            print(model.generate(sample))

        for sample in val_loader:
            print(model.generate(sample))

        print(model.parameters())

        # if loading to cpu
        # device = torch.deivce('cpu')
        # model = PolyhedronModel(n_gc_layers = 1, n_neurons=[n_node_features])
        # model.load_state_dict(torch.load(PATH,map_location=device))
    else:
        model = PolyhedronModel(n_gc_layers = 1, n_neurons=[n_node_features])
        m = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # for sample in train_loader:
        #     print(sample[2])
        #     print(model(sample,targets=sample[2] ))

        # lambda1 = lambda epoch: 0.95
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda1)
        # scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,lambda1)
        # scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=n_epochs, power=2.0)

    # Training Looop
    if train_model:
        factor = float(n_train) / float(n_validation)
        n_epoch_0 = 0
        model.train()
        for epoch in range(n_epochs):

            n_epoch = n_epoch_0 + epoch
            train_loss = 0.0
            for sample in train_loader:

                optimizer.zero_grad()
                out, loss = model(sample , targets = sample[2])
                loss.backward()
                optimizer.step()
            
                train_loss += loss.item()
            # scheduler.step()

            val_loss = 0.0
            for sample in val_loader:

                torch.set_grad_enabled(False)

                out, loss = model(sample , targets = sample[2])
                torch.set_grad_enabled(True)

                val_loss += loss.item()

            # val_running_loss *= (factor)  # to put it on the same scale as the training running loss)
            
            val_loss /= len(val_loader)
            train_loss /= len(train_loader)
            print(repr(n_epoch) + ",  " + repr(train_loss) + ",  " + repr(val_loss))

        if save_model:
            checkpoint = {
                "epoch":n_epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                }
            torch.save(checkpoint, save_model)
if __name__=='__main__':
    main()

