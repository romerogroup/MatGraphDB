import os
import shutil

import mlflow
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from parquetdb import ParquetDB
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, SAGEConv, global_mean_pool

from matgraphdb.core.datasets.mp_near_hull import MPNearHull
from matgraphdb.pyg.data.crystal_graph import CrystalGraphBuilder
from matgraphdb.pyg.models.cg_target.model import CGConvModel

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
print(torch.__version__)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name()}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cg_target_experiment(
    experiment_name="crystal_graph_torch",
    train_size_ratio=0.8,
    batch_size=32,
    hidden_channels_ratio=1,
    learning_rate=1e-3,
    num_layers=1,
    num_ffw_layers=1,
    pool_fn=global_mean_pool,
    num_epochs=100,
    use_edge_features=True,
    apply_log_to_target=False,
    loss_fn=nn.MSELoss(),
    optimizer=torch.optim.Adam,
    element_feature_keys=[
        "atomic_mass",
        "radius_covalent",
        "radius_vanderwaals",
        "heat_specific",
    ],
    target="elasticity.g_vrh",
    bond_connections_key="bonding.electric_consistent.bond_connections",
    bond_orders_key="bonding.electric_consistent.bond_orders",
    material_feature_keys=None,
    filters=None,
    interval=10,
    parameters={},
):

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment(experiment_name)
    # mlflow.autolog()
    with mlflow.start_run():

        mlflow.log_params(
            {
                "train_size_ratio": train_size_ratio,
                "batch_size": batch_size,
                "hidden_channels_ratio": hidden_channels_ratio,
                "learning_rate": learning_rate,
                "num_layers": num_layers,
                "num_ffw_layers": num_ffw_layers,
                "num_epochs": num_epochs,
                "use_edge_features": use_edge_features,
                "apply_log_to_target": apply_log_to_target,
                "loss_fn": (
                    loss_fn.__class__.__name__
                    if hasattr(loss_fn, "__class__")
                    else str(loss_fn)
                ),
                "optimizer": (
                    optimizer.__name__
                    if hasattr(optimizer, "__name__")
                    else str(optimizer)
                ),
                "element_feature_keys": element_feature_keys,
                "target": target,
                "bond_connections_key": bond_connections_key,
                "bond_orders_key": bond_orders_key,
                "material_feature_keys": material_feature_keys,
                "filters": filters,
            }
        )
        if parameters:
            mlflow.log_params(parameters)

        mdb = MPNearHull()

        builder = CrystalGraphBuilder(
            mdb,
            element_feature_keys=element_feature_keys,
            target=target,
            bond_connections_key=bond_connections_key,
            bond_orders_key=bond_orders_key,
            material_feature_keys=material_feature_keys,
            filters=filters,
            apply_log_to_target=apply_log_to_target,
        )
        data_list = builder.process_materials()

        print(f"Size of dataset: {len(data_list)}")
        train_size = int(train_size_ratio * len(data_list))  # 80% for training
        val_ratio = (1 - train_size_ratio) / 2
        val_size = int(val_ratio * len(data_list))  # 10% for validation
        # test_size = int(val_ratio * len(data_list))  # 10% for testing
        train_data = data_list[:train_size]
        val_data = data_list[train_size : train_size + val_size]
        test_data = data_list[train_size + val_size :]  # Remaining 10% for testing

        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        print("-" * 100)
        print(f"x : {data_list[0].x.shape}")
        print(f"edge_index: {data_list[0].edge_index.shape}")
        print(f"edge_attr: {data_list[0].edge_attr.shape}")
        print(f"pos: {data_list[0].pos.shape}")
        print(f"y: {data_list[0].y.shape}")
        print("-" * 100)

        num_node_features = data_list[0].x.shape[1]  # Example
        num_edge_features = data_list[0].edge_attr.shape[1]  # Example
        out_channels = data_list[0].y.shape[0]
        hidden_channels = num_node_features // hidden_channels_ratio  # Example

        if not use_edge_features:
            num_edge_features = 0

        print(f"Number of node features : {num_node_features}")
        print(f"Number of edge features : {num_edge_features}")
        print(f"Hidden channels : {hidden_channels}")
        print(f"Output channels : {out_channels}")
        print("-" * 100)

        model = CGConvModel(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_ffw_layers=num_ffw_layers,
            pool_fn=pool_fn,
        )

        model = model.to(device)

        # 2) Define your optimizer and loss function
        optimizer = optimizer(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # 4) Run multiple epochs
        for epoch in range(1, num_epochs + 1):

            if epoch % interval == 0:
                record_loss = True
            else:
                record_loss = False

            train_loss, train_log_loss = train(
                model,
                train_loader,
                optimizer,
                loss_fn,
                is_log=apply_log_to_target,
                record_loss=record_loss,
            )
            if record_loss:
                val_loss, val_log_loss = test(
                    model,
                    val_loader,
                    loss_fn,
                    is_log=apply_log_to_target,
                    record_loss=record_loss,
                )
                test_loss, test_log_loss = test(
                    model,
                    test_loader,
                    loss_fn,
                    is_log=apply_log_to_target,
                    record_loss=record_loss,
                )
                print(
                    f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}|{train_log_loss:.4f}, Val Loss: {val_loss:.4f}|{val_log_loss:.4f}, Test Loss: {test_loss:.4f}|{test_log_loss:.4f}"
                )

                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_log_loss", val_log_loss, step=epoch)
                mlflow.log_metric("test_loss", test_loss, step=epoch)
                mlflow.log_metric("test_log_loss", test_log_loss, step=epoch)

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_log_loss", train_log_loss, step=epoch)

        batch_loss, batch_log_loss, results_df = evaluate(
            model, test_loader, loss_fn, is_log=apply_log_to_target
        )

        results_csv = "predictions.csv"
        results_df.to_csv(results_csv, index=False)
        mlflow.log_artifact(results_csv)

        # -------------------------
        # 8) LOG THE MODEL
        # -------------------------
        mlflow.pytorch.log_model(model, "model")


def evaluate(model, loader, loss_fn, is_log=False):
    model.eval()
    num_batches = len(loader)
    total_loss = 0
    total_log_loss = 0
    predicitions = {
        "y_true": [],
        "y_pred": [],
        "y_true_log": [],
        "y_pred_log": [],
    }

    for data in loader:
        data = data.to(device)

        out = model(data).squeeze()

        if is_log:
            y_true_log = data.y
            y_true = 10**y_true_log
            y_pred_log = out
            y_pred = 10**y_pred_log
        else:
            y_true_log = torch.log10(data.y)
            y_true = data.y
            y_pred = out
            y_pred_log = torch.log10(y_pred)

        loss = loss_fn(out, y_true)
        total_loss += loss.item()

        log_loss = loss_fn(y_pred_log, y_true_log)
        total_log_loss += log_loss.item()

        predicitions["y_true"].extend(y_true.cpu().detach().numpy())
        predicitions["y_pred"].extend(y_pred.cpu().detach().numpy())
        predicitions["y_pred_log"].extend(y_pred_log.cpu().detach().numpy())
        predicitions["y_true_log"].extend(y_true_log.cpu().detach().numpy())

    df = pd.DataFrame(predicitions)
    df["diff"] = df["y_pred"] - df["y_true"]
    df["diff_log"] = df["y_pred_log"] - df["y_true_log"]
    df["diff2"] = df["diff"] ** 2
    df["diff2_log"] = df["diff_log"] ** 2

    batch_loss = total_loss / num_batches
    batch_log_loss = total_log_loss / num_batches
    return batch_loss, batch_log_loss, df


# 3) Training loop
def train(model, loader, optimizer, loss_fn, is_log=False, record_loss=False):
    model.train()
    num_batches = len(loader)
    total_loss = 0
    total_log_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        # forward pass
        out = model(data).squeeze()
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        if record_loss:
            if is_log:
                y_true_log = data.y
                y_true = 10**y_true_log
                y_pred_log = out
                y_pred = 10**y_pred_log
            else:
                y_true_log = torch.log10(data.y)
                y_true = data.y
                y_pred = out
                y_pred_log = torch.log10(y_pred)

            total_loss += loss_fn(y_pred, y_true).item()
            total_log_loss += loss_fn(y_pred_log, y_true_log).item()

    batch_loss = total_loss / num_batches
    batch_log_loss = total_log_loss / num_batches
    return batch_loss, batch_log_loss


def test(model, loader, loss_fn, is_log=False, record_loss=False):
    model.eval()
    total_loss = 0
    total_log_loss = 0
    predicitions = {
        "y_true": [],
        "y_pred": [],
        "y_true_log": [],
        "y_pred_log": [],
    }
    num_batches = len(loader)
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).squeeze()
            if record_loss:
                if is_log:
                    y_true_log = data.y
                    y_true = 10**y_true_log
                    y_pred_log = out
                    y_pred = 10**y_pred_log
                else:
                    y_true_log = torch.log10(data.y)
                    y_true = data.y
                    y_pred = out
                    y_pred_log = torch.log10(y_pred)

                total_loss += loss_fn(y_pred, y_true).item()
                total_log_loss += loss_fn(y_pred_log, y_true_log).item()
    batch_loss = total_loss / num_batches
    batch_log_loss = total_log_loss / num_batches
    return batch_loss, batch_log_loss
