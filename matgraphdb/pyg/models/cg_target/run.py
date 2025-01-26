import pyarrow as pa
import pyarrow.compute as pc
import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, global_mean_pool

from matgraphdb import config
from matgraphdb.materials.datasets.mp_near_hull import MPNearHull
from matgraphdb.pyg.core import BaseTrainer
from matgraphdb.pyg.core.experiment import run_experiment
from matgraphdb.pyg.data import CrystalGraphBuilder
from matgraphdb.pyg.models.cg_target.model import CGConvModel

print(torch.__version__)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name()}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config.logging_config.loggers.matgraphdb.level = "INFO"
config.apply()

data_config = {
    "element_feature_keys": [
        "atomic_mass",
        "radius_covalent",
        "radius_vanderwaals",
        "heat_specific",
    ],
    "target": "elasticity.g_vrh",
    "bond_connections_key": "bonding.electric_consistent.bond_connections",
    "bond_orders_key": "bonding.electric_consistent.bond_orders",
    "material_feature_keys": None,
    "material_filters": [
        pc.field("elasticity.g_vrh") > 0,
        pc.field("elasticity.g_vrh") < 400,
    ],
    "apply_log_to_target": True,
}
mdb = MPNearHull()
builder = CrystalGraphBuilder(
    mdb,
    element_feature_keys=data_config.get("element_feature_keys"),
    target=data_config.get("target"),
    bond_connections_key=data_config.get("bond_connections_key"),
    bond_orders_key=data_config.get("bond_orders_key"),
    material_feature_keys=data_config.get("material_feature_keys"),
    material_filters=data_config.get("material_filters"),
    apply_log_to_target=data_config.get("apply_log_to_target"),
)
data_list = builder.build()
num_node_features = data_list[0].x.shape[1]  # Example
num_edge_features = data_list[0].edge_attr.shape[1]  # Example
out_channels = data_list[0].y.shape[0]
hidden_channels = num_node_features

print(f"Number of node features : {num_node_features}")
print(f"Number of edge features : {num_edge_features}")
print(f"Hidden channels : {hidden_channels}")
print(f"Output channels : {out_channels}")

model_config = {
    "num_node_features": num_node_features,
    "num_edge_features": num_edge_features,
    "out_channels": out_channels,
    "hidden_channels": 32,
    "num_layers": 2,
    "num_ffw_layers": 1,
    "pool_fn": global_mean_pool,
}


experiment_config = {
    "experiment_name": "cg_target_experiment",
}
train_config = {
    "optimizer": torch.optim.Adam,
    "loss_fn": nn.MSELoss(),
    "num_epochs": 10,
    "interval": 10,
    "train_size_ratio": 0.8,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "seed": 42,
    "device": device,
    "optimizer_kwargs": {"lr": 1e-3},
}


run_experiment(
    model_cls=CGConvModel,
    trainer_cls=BaseTrainer,
    data_list=data_list,
    experiment_config=experiment_config,
    model_config=model_config,
    train_config=train_config,
    data_config=data_config,
)
