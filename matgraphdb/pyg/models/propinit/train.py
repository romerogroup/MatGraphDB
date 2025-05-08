import copy
import json
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch_geometric as pyg
import torch_geometric.transforms as T
from omegaconf import OmegaConf
from pyg_lib.partition import metis

# from matgraphdb.pyg.models.heterograph_encoder_general.model import MaterialEdgePredictor
# from matgraphdb.pyg.models.heterograph_encoder_general.trainer import (
#     Trainer,
#     learning_curve,
#     pca_plots,
#     roc_curve,
# )
# from sklearn.metrics import (
#     mean_absolute_error,
#     mean_squared_error,
#     r2_score,
#     roc_auc_score,
#     roc_curve,
# )
from sklearn import linear_model
from torch_geometric import nn as pyg_nn
from torch_geometric.index import index2ptr, ptr2index
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor

from matgraphdb.core.datasets.mp_near_hull import MPNearHull
from matgraphdb.pyg.builders import HeteroGraphBuilder
from matgraphdb.pyg.models.heterograph_encoder_general.metrics import (
    LearningCurve,
    ROCCurve,
    plot_pca,
)
from matgraphdb.pyg.models.propinit.model import Model

########################################################################################################################

DATA_CONFIG = OmegaConf.create(
    {
        "dataset_dir": os.path.join("data", "datasets", "MPNearHull"),
        "create_random_features": False,
        "n_material_dim": 4,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
    }
)

PROPINET_CONFIG = OmegaConf.create(
    {
        "data": dict(DATA_CONFIG),
        "model": {
            "out_channels": 1,
            "out_node_type": "materials",
            "hidden_channels": 32,
            "n_partitions": 4,
            "k_steps": 3,
            "use_projections": True,
            "use_embeddings": False,
            "use_shallow_embedding_for_materials": False,
            "use_projections_for_materials": True,
        },
        "training": {
            "training_dir": os.path.join("data", "training_runs", "propinit"),
            "learning_rate": 0.01,
            "num_epochs": 501,
            "eval_interval": 100,
            "scheduler_milestones": [50],
        },
    }
)


MLP_CONFIG = OmegaConf.create(
    {
        "data": dict(DATA_CONFIG),
        "model": {
            "mlp_hidden_dim": 32,  # hidden dimension for the MLP baseline
        },
        "training": {
            "learning_rate": 0.001,
            "epochs": 4000,
        },
    }
)

LINEAR_CONFIG = OmegaConf.create(
    {
        "data": dict(DATA_CONFIG),
        "model": {
            "linear_hidden_dim": 32,  # hidden dimension for the linear baseline
        },
        "training": {
            "learning_rate": 0.001,
            "epochs": 4000,
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
def build_heterograph():
    """Build the initial heterogeneous graph from the materials database.

    Returns:
        torch_geometric.data.HeteroData: The constructed heterogeneous graph
    """
    mdb = MPNearHull(DATA_CONFIG.dataset_dir)
    builder = HeteroGraphBuilder(mdb)

    # Define the "materials" node type (only a subset of columns is used here)
    builder.add_node_type(
        "materials",
        columns=[
            "core.density_atomic",
        ],
    )

    # Define additional node types.
    builder.add_node_type(
        "elements",
        columns=[
            "atomic_mass",
            "radius_covalent",
            "radius_vanderwaals",
        ],
        drop_null=True,
        label_column="symbol",
    )
    builder.add_node_type("space_groups", drop_null=True, label_column="spg")
    builder.add_node_type(
        "crystal_systems", drop_null=True, label_column="crystal_system"
    )

    # Define edge types.
    builder.add_edge_type("element_element_neighborsByGroupPeriod")
    builder.add_edge_type("material_element_has")
    builder.add_edge_type("material_spg_has")
    builder.add_edge_type("material_crystalSystem_has")

    # Define a helper function for target encoding.
    def to_log(x):
        return torch.tensor(np.log10(x), dtype=torch.float32)

    # Add a target property for the "materials" node type.
    builder.add_target_node_property(
        "materials",
        columns=["elasticity.g_vrh"],
        filters=[
            pc.field("elasticity.g_vrh") > 0,
            pc.field("elasticity.g_vrh") < 400,
        ],
        encoders={"elasticity.g_vrh": to_log},
    )
    heterodata = builder.hetero_data
    heterodata["materials"].original_x = heterodata[
        "materials"
    ].x  # Save original features

    # Optionally, create random features if desired.
    if DATA_CONFIG.create_random_features:
        n_materials = heterodata["materials"].num_nodes
        heterodata["materials"].x = torch.normal(
            mean=0.0, std=1.0, size=(n_materials, DATA_CONFIG.n_material_dim)
        )

    return heterodata


def partition_and_convert_graph(source_data):
    """Convert heterogeneous graph to homogeneous, partition it, and convert back.

    Args:
        source_data: The source heterogeneous graph data
        config: Configuration object containing model parameters

    Returns:
        heterodata: The processed heterogeneous graph with partitioning
    """
    homodata = source_data.to_homogeneous()
    rowptr = index2ptr(homodata.edge_index[0])
    col = homodata.edge_index[1]
    node_partitions = metis(
        rowptr=rowptr, col=col, num_partitions=PROPINET_CONFIG.model.n_partitions
    )
    homodata.partition = node_partitions
    homodata.node_type_id = homodata.node_type
    heterodata = homodata.to_heterogeneous()

    # Transfer target information and feature vectors.
    heterodata["materials"].y = heterodata.y
    heterodata["materials"].y_index = heterodata.y_index
    heterodata["materials"].target_feature_mask = heterodata.target_feature_mask

    node_types = source_data.metadata()[0]
    for nt in node_types:
        if hasattr(source_data[nt], "x"):
            heterodata[nt].x = source_data[nt].x

    return heterodata


def split_by_material_nodes(parent_data):
    """Split material nodes into train/val/test sets and create corresponding subgraphs.

    Args:
        parent_data: The full heterograph containing all data
        config: Configuration object containing split ratios

    Returns:
        Dictionary containing the split subgraphs for train, validation and test sets
    """
    # Split the "materials" nodes into train/val/test
    n_materials = parent_data["materials"].num_nodes
    node_ids = parent_data["materials"].node_ids
    material_indices = torch.randperm(n_materials)

    train_ratio = DATA_CONFIG.train_ratio
    val_ratio = DATA_CONFIG.val_ratio
    test_ratio = 1 - train_ratio

    train_size = int(train_ratio * n_materials)
    test_size = int(test_ratio * n_materials)
    train_val_size = int(val_ratio * train_size)
    test_val_size = int(val_ratio * test_size)

    total_train_materials = material_indices[:train_size]
    total_test_materials = material_indices[train_size:]

    # Split train and test into their validation sets
    train_val_materials = total_train_materials[:train_val_size]
    train_materials = total_train_materials[train_val_size:]
    test_val_materials = total_test_materials[:test_val_size]
    test_materials = total_test_materials[test_val_size:]

    print("\nSplit percentages:")
    print(f"Total: {n_materials}")
    print(f"Train: {len(train_materials)/n_materials*100:.1f}%")
    print(f"Train val: {len(train_val_materials)/n_materials*100:.1f}%")
    print(f"Test: {len(test_materials)/n_materials*100:.1f}%")
    print(f"Test val: {len(test_val_materials)/n_materials*100:.1f}%")
    total_pct = (
        (
            len(train_materials)
            + len(train_val_materials)
            + len(test_materials)
            + len(test_val_materials)
        )
        / n_materials
        * 100
    )
    print(f"Total: {total_pct:.1f}%\n")

    # Create subgraphs for each split
    split_dicts = {
        "train": {"materials": train_materials},
        "train_val": {"materials": train_val_materials},
        "test": {"materials": test_materials},
        "test_val": {"materials": test_val_materials},
    }

    split_data = {}
    for split_name, split_dict in split_dicts.items():
        data = parent_data.subgraph(split_dict)
        data["materials"].node_ids = parent_data["materials"].node_ids[
            split_dict["materials"]
        ]
        split_data[split_name] = data

    print(split_data["train"]["materials"].node_ids)
    print(f"Train materials: {len(train_materials)}")
    print(f"Train val materials: {len(train_val_materials)}")
    print(f"Test materials: {len(test_materials)}")
    print(f"Test val materials: {len(test_val_materials)}")

    # For each split, reduce the target values and record indices
    y_id_map = {
        int(y_id): float(y)
        for y_id, y in zip(parent_data["materials"].y_index, parent_data["materials"].y)
    }

    for data in split_data.values():
        y_vals = []
        ids = []
        node_ids_list = []
        for i, node_id in enumerate(data["materials"].node_ids):
            if int(node_id) in y_id_map:
                y_vals.append(y_id_map[int(node_id)])
                node_ids_list.append(node_id)
                ids.append(i)
        data["materials"].y = torch.tensor(y_vals)
        data["materials"].y_node_ids = torch.tensor(node_ids_list)
        data["materials"].y_split_index = torch.tensor(ids)

    return split_data


def heterograph_preprocessing():
    """
    Build the heterograph, apply transformations, partition the graph,
    and split the 'materials' nodes into training/validation/test subgraphs.

    Args:
        config (OmegaConf): A configuration object with the keys:
            - data: data-related parameters (e.g., dataset_dir, create_random_features, n_material_dim, train_ratio, val_ratio)
            - model: model-related parameters (e.g., n_partitions)
            - training: training-related parameters

    Returns:
        split_data (dict): A dictionary with keys "train", "train_val", "test", "test_val",
                           each containing a subgraph for the corresponding split.
    """
    # 1. Build the heterogeneous graph from the materials database

    original_heterograph = build_heterograph()

    # 2. Apply transformation: make the graph undirected.
    source_data = T.ToUndirected()(original_heterograph)
    # Free up memory.
    original_heterograph = None

    # (edge_types not used further here but available as metadata[1])

    heterodata = partition_and_convert_graph(source_data)
    split_data = split_by_material_nodes(heterodata)

    return split_data, heterodata


def learning_curve(
    metrics_per_split, metric_name, epoch_save_path=None, total_save_path=None
):
    learning_curve = LearningCurve()

    for split_label, metrics_dict in metrics_per_split.items():
        epochs = metrics_dict["epochs"]
        loss_values = metrics_dict[metric_name]

        # Add main model curve
        learning_curve.add_curve(
            epochs, loss_values, split_label, split_label, is_baseline=False
        )

    # Generate and save plots
    learning_curve.plot()
    learning_curve.set_ylabel(metric_name)
    if epoch_save_path is not None:
        learning_curve.save(epoch_save_path)
    if total_save_path is not None:
        learning_curve.save(total_save_path)
    learning_curve.close()


########################################
# 2. Model Definitions
########################################
class LinearBaseline(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearBaseline, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class MLPBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(MLPBaseline, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_linear_baseline(split_data):
    """Train a simple linear model using PyTorch."""
    input_dim = split_data["train"]["materials"].original_x.shape[1]
    model = LinearBaseline(input_dim=input_dim)
    optimizer = optim.Adam(model.parameters(), lr=LINEAR_CONFIG.training.learning_rate)
    loss_fn = nn.L1Loss()

    # Initialize results storage
    baseline_results = {
        "linear": {
            "train": {"loss": [], "mae": [], "epochs": []},
            "train_val": {"loss": [], "mae": [], "epochs": []},
            "test": {"loss": [], "mae": [], "epochs": []},
            "test_val": {"loss": [], "mae": [], "epochs": []},
        }
    }

    num_epochs = LINEAR_CONFIG.training.epochs
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        # Training step on the training split
        x = split_data["train"]["materials"].original_x[
            split_data["train"]["materials"].y_split_index
        ]
        y = split_data["train"]["materials"].y
        y_pred = model(x).squeeze()
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        # Evaluate on all splits
        with torch.no_grad():
            for split, data_batch in split_data.items():
                model.eval()
                x_eval = data_batch["materials"].original_x[
                    data_batch["materials"].y_split_index
                ]
                y_eval = data_batch["materials"].y
                y_pred_eval = model(x_eval).squeeze()
                # Convert predictions and targets from log-scale to original scale (if applicable)
                y_pred_orig = 10 ** y_pred_eval.cpu().numpy()
                y_true_orig = 10 ** y_eval.cpu().numpy()
                mae = np.mean(np.abs(y_pred_orig - y_true_orig))
                loss = loss_fn(y_pred_eval, y_eval)
                baseline_results["linear"][split]["loss"].append(loss)
                baseline_results["linear"][split]["mae"].append(mae)
                baseline_results["linear"][split]["epochs"].append(epoch)

    print(f"[Linear] Epoch {epoch:3d} | Loss: {loss.item():.4f} | MAE: {mae}")

    return baseline_results["linear"]


def train_mlp_baseline(split_data):
    """Train an MLP baseline model using PyTorch."""
    input_dim = split_data["train"]["materials"].original_x.shape[1]
    hidden_dim = MLP_CONFIG.model.mlp_hidden_dim
    model = MLPBaseline(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=MLP_CONFIG.training.learning_rate)
    loss_fn = nn.L1Loss()

    # Initialize results storage
    baseline_results = {
        "mlp": {
            "train": {"loss": [], "mae": [], "epochs": []},
            "train_val": {"loss": [], "mae": [], "epochs": []},
            "test": {"loss": [], "mae": [], "epochs": []},
            "test_val": {"loss": [], "mae": [], "epochs": []},
        }
    }

    num_epochs = MLP_CONFIG.training.epochs
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        # Training step on the training split
        x = split_data["train"]["materials"].original_x[
            split_data["train"]["materials"].y_split_index
        ]
        y = split_data["train"]["materials"].y
        y_pred = model(x).squeeze()
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        # Evaluate on all splits
        with torch.no_grad():
            for split, data_batch in split_data.items():
                model.eval()
                x_eval = data_batch["materials"].original_x[
                    data_batch["materials"].y_split_index
                ]
                y_eval = data_batch["materials"].y
                y_pred_eval = model(x_eval).squeeze()
                y_pred_orig = 10 ** y_pred_eval.cpu().numpy()
                y_true_orig = 10 ** y_eval.cpu().numpy()
                mae = np.mean(np.abs(y_pred_orig - y_true_orig))

                loss = loss_fn(y_pred_eval, y_eval)

                baseline_results["mlp"][split]["loss"].append(loss)
                baseline_results["mlp"][split]["mae"].append(mae)
                baseline_results["mlp"][split]["epochs"].append(epoch)

    print(f"[MLP] Epoch {epoch:3d} | Loss: {loss.item():.4f} | MAE: {mae}")

    return baseline_results["mlp"]


def train_propinet(split_data, heterodata):
    """
    Train the Propinet model using the given heterograph and data splits.

    The function trains the model for a specified number of epochs,
    evaluates performance (MAE and RMSE) on each split (train_val, test, test_val),
    and records the training loss for the train split.

    Args:
        split_data (dict): Dictionary with keys "train", "train_val", "test", "test_val"
                           containing the corresponding subgraphs.
        heterodata: The full heterogeneous graph data.

    Returns:
        model: The trained model.
        results (dict): A dictionary recording the metrics for each epoch. The schema is:
            {
                "train":     {"loss": [list of train losses], "epochs": [list of epochs]},
                "train_val": {"mae":  [list of MAE], "rmse": [list of RMSE], "epochs": [list of epochs]},
                "test":      {"mae":  [list of MAE], "rmse": [list of RMSE], "epochs": [list of epochs]},
                "test_val":  {"mae":  [list of MAE], "rmse": [list of RMSE], "epochs": [list of epochs]}
            }
    """
    # Initialize the model using configuration parameters.
    model = Model(
        out_channels=PROPINET_CONFIG.model.out_channels,
        out_node_type=PROPINET_CONFIG.model.out_node_type,
        hidden_channels=PROPINET_CONFIG.model.hidden_channels,
        heterodata=heterodata,
        use_projections=PROPINET_CONFIG.model.use_projections,
        use_embeddings=PROPINET_CONFIG.model.use_embeddings,
        use_shallow_embedding_for_materials=PROPINET_CONFIG.model.use_shallow_embedding_for_materials,
        use_projections_for_materials=PROPINET_CONFIG.model.use_projections_for_materials,
        n_partitions=PROPINET_CONFIG.model.n_partitions,
        k_steps=PROPINET_CONFIG.model.k_steps,
    )
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=PROPINET_CONFIG.training.learning_rate
    )
    loss_fn = nn.L1Loss()

    def train_step(data_batch):
        """
        Performs a single training step on the provided batch.
        """
        model.train()
        optimizer.zero_grad()

        out = model(data_batch).squeeze()
        y_split_index = data_batch["materials"].y_split_index
        y_target = data_batch["materials"].y
        y_pred = out[y_split_index]

        loss = loss_fn(y_pred, y_target)
        loss.backward()
        optimizer.step()
        return float(loss.cpu())

    @torch.no_grad()
    def evaluation_step(data_batch):
        """
        Evaluates the model on the provided batch and computes MAE and RMSE.
        """
        model.eval()
        out = model(data_batch).squeeze()
        y_split_index = data_batch["materials"].y_split_index
        y_target = data_batch["materials"].y
        y_pred = out[y_split_index]
        loss = loss_fn(y_pred, y_target)

        y_pred = y_pred.cpu().numpy()
        y_target = y_target.cpu().numpy()
        # Convert predictions from log-scale back to original scale.
        y_pred_orig = 10**y_pred
        y_target_orig = 10**y_target

        mae = np.mean(np.abs(y_pred_orig - y_target_orig))

        return {"loss": loss, "mae": mae}

    # Initialize the results dictionary.
    results = {
        "train": {"loss": [], "mae": [], "epochs": []},
        "train_val": {"loss": [], "mae": [], "epochs": []},
        "test": {"loss": [], "mae": [], "epochs": []},
        "test_val": {"loss": [], "mae": [], "epochs": []},
    }

    num_epochs = PROPINET_CONFIG.training.num_epochs
    for epoch in range(num_epochs):
        # Perform one training step on the training split.
        train_loss = train_step(split_data["train"])
        results["train"]["loss"].append(train_loss)
        results["train"]["epochs"].append(epoch)

        # Evaluate model on the evaluation splits.
        for split in ["train", "train_val", "test", "test_val"]:
            metrics = evaluation_step(split_data[split])
            results[split]["mae"].append(metrics["mae"])
            results[split]["loss"].append(metrics["loss"])
            results[split]["epochs"].append(epoch)

        # scheduler.step()

        # Print progress at the configured evaluation interval.
        if epoch % PROPINET_CONFIG.training.eval_interval == 0:
            print(
                f"Epoch {epoch+1:4d} | Train Loss: {train_loss:.4f} | "
                f"Train_val Loss: {results['train_val']['loss'][-1]:.4f} | "
                f"Test Loss: {results['test']['loss'][-1]:.4f} | "
                f"Test_val Loss: {results['test_val']['loss'][-1]:.4f}"
            )

    return results


def main():

    # Create dummy splits: "train", "train_val", "test", "test_val"
    split_data, heterodata = heterograph_preprocessing()

    # Run the separate training loops
    print("Training Linear Baseline...")
    linear_results = train_linear_baseline(split_data)

    print("Training MLP Baseline...")
    mlp_results = train_mlp_baseline(split_data)

    print("Training Propinet...")
    propinet_results = train_propinet(split_data, heterodata)

    runs_dir = os.path.join("data", "training_runs", "propinit", "runs")
    os.makedirs(runs_dir, exist_ok=True)

    n_runs = len(os.listdir(runs_dir))
    run_dir = os.path.join(runs_dir, f"run_{n_runs+1}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(
            {
                "linear": linear_results,
                "mlp": mlp_results,
                "propinet": propinet_results,
            },
            f,
        )

    # Plot all learning curves using the LearningCurve class
    learning_curve_plot = LearningCurve()
    for model_name, results in [
        ("linear", linear_results),
        ("mlp", mlp_results),
        ("propinet", propinet_results),
    ]:
        for split, metrics in results.items():
            label = f"{model_name}-{split}"
            learning_curve_plot.add_curve(
                metrics["epochs"], metrics["mae"], label, label, is_baseline=True
            )
    learning_curve_plot.plot()
    learning_curve_plot.save(os.path.join(run_dir, "learning_curve.png"))
    learning_curve_plot.close()


if __name__ == "__main__":
    main()
