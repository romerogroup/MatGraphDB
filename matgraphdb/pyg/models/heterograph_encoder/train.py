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
from matgraphdb.pyg.data import HeteroGraphBuilder
from matgraphdb.pyg.models.heterograph_encoder.model import MaterialEdgePredictor
from matgraphdb.pyg.models.heterograph_encoder.trainer import (
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
            "n_material_dim": 4,
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
            "hidden_channels": 32,
            "num_decoder_layers": 2,
            "num_conv_layers": 3,
            "num_ffw_layers": 3,
            "dropout_rate": 0.0,
            "use_projections": True,
            "use_embeddings": True,
            "use_shallow_embedding_for_materials": False,
        },
        "training": {
            "training_dir": os.path.join(
                "data", "training_runs", "heterograph_encoder"
            ),
            "learning_rate": 0.001,
            "num_epochs": 10001,
            "eval_interval": 1000,
            "weights": None,
            "use_weights": False,
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

# n_materials = data["materials"].num_nodes
# material_edge_index = torch.tensor(
#     np.array([np.arange(n_materials), np.arange(n_materials)]),
#     dtype=torch.int64
# )
# data["materials", "connectsSelf", "materials"].edge_index = material_edge_index
# flip_edge_index = torch.flip(data["materials", "has", "elements"].edge_index, dims=[0])
# data["elements", "occursIn", "materials"].edge_index = torch.flip(
#     data["materials", "has", "elements"].edge_index, dims=[0]
# )

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

print(CONFIG.data.random_link_split_args)

random_link_split_args = OmegaConf.to_container(
    CONFIG.data.random_link_split_args, resolve=True
)
print(type(random_link_split_args))
for i, edge_type in enumerate(random_link_split_args["edge_types"]):
    random_link_split_args["edge_types"][i] = tuple(edge_type)

for i, edge_type in enumerate(random_link_split_args["rev_edge_types"]):
    random_link_split_args["rev_edge_types"][i] = tuple(edge_type)

print(random_link_split_args)
# Perform a link-level split into training, validation, and test edges:
train_data, _, _ = T.RandomLinkSplit(**random_link_split_args)(original_train_data)
train_val_data, _, _ = T.RandomLinkSplit(**random_link_split_args)(
    original_train_val_data
)
test_data, _, _ = T.RandomLinkSplit(**random_link_split_args)(original_test_data)
test_val_data, _, _ = T.RandomLinkSplit(**random_link_split_args)(
    original_test_val_data
)


print(train_data)
print(train_val_data)
print(test_data)
print(test_val_data)

train_data = train_data.to(device)
train_val_data = train_val_data.to(device)
test_data = test_data.to(device)
test_val_data = test_val_data.to(device)

print(test_val_data)

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


def compute_class_distribution(data):
    edge_label = data["materials", "elements"].edge_label.to(torch.int64)
    class_count = torch.bincount(edge_label)
    unique_classes = torch.unique(edge_label)
    majority_class = int(torch.argmax(class_count).item())
    proposed_weights = class_count.max() / class_count

    print(f"Unique classes: {unique_classes}")
    print(f"Class_count: {class_count}")
    print(f"Proposed weights: {proposed_weights}")
    print(f"Majority class: {majority_class}")

    return edge_label, majority_class, proposed_weights


print(f"Classes: Edge does not exist (0) | Edge exists (1) ")

for label in split_data.keys():
    data = split_data[label]["data"]
    edge_label, majority_class, proposed_weights = compute_class_distribution(data)

    split_data[label]["edge_label"] = edge_label
    split_data[label]["majority_class"] = majority_class
    split_data[label]["proposed_weights"] = proposed_weights

weights = CONFIG.training.weights
if weights is None and CONFIG.training.use_weights:
    weights = split_data["train"]["proposed_weights"]


print(f"Actual Weights Used: {weights}")
print("-" * 100)

####################################################################################################
####################################################################################################
####################################################################################################
# Model
####################################################################################################
model = MaterialEdgePredictor(
    hidden_channels=CONFIG.model.hidden_channels,
    data=parent_data,
    decoder_kwargs={
        "num_layers": CONFIG.model.num_decoder_layers,
        "dropout": CONFIG.model.dropout_rate,
        "src_node_name": "materials",
        "tgt_node_name": "elements",
    },
    encoder_kwargs={
        "num_conv_layers": CONFIG.model.num_conv_layers,
        "num_ffw_layers": CONFIG.model.num_ffw_layers,
        "dropout": CONFIG.model.dropout_rate,
    },
    use_projections=CONFIG.model.use_projections,
    use_embeddings=CONFIG.model.use_embeddings,
    use_shallow_embedding_for_materials=CONFIG.model.use_shallow_embedding_for_materials,
).to(device)
print(model)


####################################################################################################
# Training
####################################################################################################


def weighted_binary_cross_entropy(pred, target, weights=None):
    if weights is None:
        weights = 1.0
        y = F.binary_cross_entropy(pred, target)
        return y
    else:
        weights = weights[target.long()]
        return F.binary_cross_entropy(pred, target, weight=weights)


optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.training.learning_rate)

scheduler = lr_scheduler.MultiStepLR(
    optimizer, milestones=CONFIG.training.scheduler_milestones, gamma=0.1, verbose=False
)
# scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=10)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                            factor=0.1,
#                                            patience=100,
#                                            threshold=0.0001,
#                                            min_lr=0.00001,
#                                            verbose=False)


trainer = Trainer(
    model=model,
    loss_fn=weighted_binary_cross_entropy,
    optimizer=optimizer,
    train_data=train_data,
    train_val_data=train_val_data,
    test_data=test_data,
    test_val_data=test_val_data,
    num_epochs=CONFIG.training.num_epochs,
    eval_interval=CONFIG.training.eval_interval,
    training_dir=os.path.join("data", "training_runs", "heterograph_encoder"),
    scheduler=scheduler,
    evaluation_callbacks=[learning_curve, roc_curve, pca_plots],
    use_mlflow=CONFIG.training.use_mlflow,
    mlflow_experiment_name=CONFIG.training.mlflow_experiment_name,
    mlflow_tracking_uri=CONFIG.training.mlflow_tracking_uri,
    mlflow_record_system_metrics=CONFIG.training.mlflow_record_system_metrics,
)


trainer.train(metrics_to_record=["loss", "accuracy", "precision", "recall"])


out = model.encode(
    test_val_data.x_dict,
    test_val_data.edge_index_dict,
    node_ids={
        "materials": test_val_data["materials"].node_ids,
        "elements": test_val_data["elements"].node_ids,
    },
)


print(out)
