import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch_geometric as pyg
import torch_geometric.transforms as T
from omegaconf import OmegaConf
from torch_geometric import nn as pyg_nn

from matgraphdb.core.datasets.mp_near_hull import MPNearHull
from matgraphdb.pyg.builders import HeteroGraphBuilder
from matgraphdb.pyg.models.hgt.model import HGT
from matgraphdb.pyg.models.hgt.trainer import (
    Trainer,
    learning_curve,
    pca_plots,
    roc_curve,
)

########################################################################################################################


CONFIG = OmegaConf.create(
    {
        "data": {
            "dataset_dir": os.path.join("data", "datasets", "MPNearHull"),
            "create_random_features": True,
            "n_material_dim": 16,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "random_link_split_args": {
                "num_val": 0.0,
                "num_test": 0.0,
                "neg_sampling_ratio": 1.0,
                "is_undirected": True,
                "edge_types": [("materials", "has", "elements")],
                "rev_edge_types": [("elements", "rev_has", "materials")],
            },
        },
        "model": {
            "hidden_channels": 128,
            "out_channels": 1,
            "num_heads": 8,
            "num_layers": 3,
        },
        "training": {
            "training_dir": os.path.join(
                "data", "training_runs", "heterograph_encoder"
            ),
            "learning_rate": 0.001,
            "num_epochs": 40001,
            "eval_interval": 2000,
            "weights": None,
            "use_weights": False,
            "use_scheduler": False,
            "metrics_to_print": ["accuracy", "precision", "recall"],
            "scheduler_milestones": [4000, 20000],
            "use_mlflow": False,
            "mlflow_experiment_name": "heterograph_encoder",
            "mlflow_tracking_uri": "${training.training_dir}/mlflow",
            "mlflow_record_system_metrics": True,
        },
    }
)


####################################################################################################
####################################################################################################

print("-" * 100)
print(f"Torch version: {torch.__version__}")
print(f"PyTorch Geometric version: {pyg.__version__}")
print(
    f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")
print("-" * 100)


####################################################################################################q
# Data Loading
####################################################################################################
mdb = MPNearHull(CONFIG.data.dataset_dir)
current_dir = os.path.dirname(os.path.abspath(__file__))

builder = HeteroGraphBuilder(mdb)

builder.add_node_type(
    "materials",
    columns=[
        # "core.volume",
        "core.density",
        # "core.density_atomic",
        # "core.nelements",
        # "core.nsites",
    ],
    # embedding_vectors=True
)


def binning(x):
    return torch.tensor(np.log10(x), dtype=torch.float32)


builder.add_node_type(
    "elements",
    columns=[
        "atomic_mass",
        "radius_covalent",
        "radius_vanderwaals",
        # "electron_affinity"
        # "heat_specific",
    ],
    drop_null=True,
    # embedding_vectors=True,
    label_column="symbol",
    encoders={"electron_affinity": binning},
)

builder.add_edge_type("element_element_neighborsByGroupPeriod")
builder.add_edge_type(
    "material_element_has",
    #   columns=["weight"]
)


def to_log(x):
    return torch.tensor(np.log10(x), dtype=torch.float32)


target_is_log = True
builder.add_target_node_property(
    "materials",
    columns=["elasticity.g_vrh"],
    filters=[
        pc.field("elasticity.g_vrh") > 0,
        pc.field("elasticity.g_vrh") < 400,
    ],
    encoders={"elasticity.g_vrh": to_log},
)
data = builder.hetero_data


# Set random feature vector for materials
if CONFIG.data.create_random_features:
    n_materials = data["materials"].num_nodes
    data["materials"].x = torch.normal(
        mean=0.0, std=1.0, size=(n_materials, CONFIG.data.n_material_dim)
    )


parent_data = T.ToUndirected()(data)

print(parent_data)
data = None


####################################################################################################
# Data Splitting
####################################################################################################

# Split materials into train/val/test sets
n_materials = parent_data["materials"].num_nodes
material_indices = torch.randperm(n_materials)

# Hyperparameters for data splitting
train_ratio = CONFIG.data.train_ratio  # 80% for training set (including validation)
test_ratio = 1 - train_ratio  # 20% for test set (including validation)
val_ratio = CONFIG.data.val_ratio  # 10% validation ratio within each set


# Calculate sizes for each split
train_size = int(train_ratio * n_materials)  # Training set size
test_size = int(test_ratio * n_materials)  # Test set size
train_val_size = int(val_ratio * train_size)  # Training validation set size
test_val_size = int(val_ratio * test_size)  # Test validation set size

total_train_materials = material_indices[:train_size]
total_test_materials = material_indices[train_size:]

train_materials = total_train_materials[train_val_size:]
train_val_materials = total_train_materials[:train_val_size]
test_materials = total_test_materials[test_val_size:]
test_val_materials = total_test_materials[:test_val_size]

n_train = len(train_materials)
# Print percentages of each split
print("\nSplit percentages:")
print(f"Total: {n_materials}")
print(f"Train: {len(train_materials)/n_materials*100:.1f}%")
print(f"Train val: {len(train_val_materials)/n_materials*100:.1f}%")
print(f"Test: {len(test_materials)/n_materials*100:.1f}%")
print(f"Test val: {len(test_val_materials)/n_materials*100:.1f}%")
print(
    f"Total: {(len(train_materials) + len(train_val_materials) + len(test_materials) + len(test_val_materials))/n_materials*100:.1f}%\n"
)


# Create subgraphs for each split
train_dict = {"materials": train_materials}
train_val_dict = {"materials": train_val_materials}
test_dict = {"materials": test_materials}
test_val_dict = {"materials": test_val_materials}

original_train_data = parent_data.subgraph(train_dict)
original_train_data["materials"].node_ids = train_dict["materials"]
original_train_val_data = parent_data.subgraph(train_val_dict)
original_train_val_data["materials"].node_ids = train_val_dict["materials"]
original_test_data = parent_data.subgraph(test_dict)
original_test_data["materials"].node_ids = test_dict["materials"]
original_test_val_data = parent_data.subgraph(test_val_dict)
original_test_val_data["materials"].node_ids = test_val_dict["materials"]

print(f"Train materials: {len(train_materials)}")
print(f"Train val materials: {len(train_val_materials)}")
print(f"Test materials: {len(test_materials)}")
print(f"Test val materials: {len(test_val_materials)}")

# data = None
builder = None


train_data = original_train_data
train_val_data = original_train_val_data
test_data = original_test_data
test_val_data = original_test_val_data


print(train_data)
print(train_val_data)
print(test_data)
print(test_val_data)

train_data = train_data.to(device)
train_val_data = train_val_data.to(device)
test_data = test_data.to(device)
test_val_data = test_val_data.to(device)


data_dict = {
    "train": train_data,
    "train_val": train_val_data,
    "test": test_data,
    "test_val": test_val_data,
}

split_data = {label: {"data": data} for label, data in data_dict.items()}
print("-" * 100)
print(f"Max memory allocated: {torch.cuda.max_memory_allocated()}")


####################################################################################################
# Model
####################################################################################################
model = HGT(
    hidden_channels=CONFIG.model.hidden_channels,
    out_channels=CONFIG.model.out_channels,
    num_heads=CONFIG.model.num_heads,
    num_layers=CONFIG.model.num_layers,
    data=train_data,
).to(device)
print(model)


####################################################################################################
# Training
####################################################################################################


with torch.no_grad():  # Initialize lazy modules.
    out = model(train_data.x_dict, train_data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data["author"].train_mask
    loss = F.cross_entropy(out[mask], data["author"].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)

    accs = []
    for split in ["train_mask", "val_mask", "test_mask"]:
        mask = data["author"][split]
        acc = (pred[mask] == data["author"].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs


for epoch in range(1, 101):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(
        f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, "
        f"Val: {val_acc:.4f}, Test: {test_acc:.4f}"
    )
