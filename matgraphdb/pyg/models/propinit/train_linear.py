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
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch_geometric as pyg
import torch_geometric.transforms as T
from omegaconf import OmegaConf
from sklearn import linear_model
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from torch_geometric import nn as pyg_nn

from matgraphdb.core.datasets.mp_near_hull import MPNearHull
from matgraphdb.pyg.data import HeteroGraphBuilder
from matgraphdb.pyg.models.heterograph_encoder_general.metrics import (
    LearningCurve,
    ROCCurve,
    plot_pca,
)
from matgraphdb.pyg.models.heterograph_encoder_general.model import (
    MaterialEdgePredictor,
)
from matgraphdb.pyg.models.heterograph_encoder_general.trainer import (
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
                "edge_types": [
                    ("materials", "has", "elements"),
                    ("materials", "has", "space_groups"),
                    ("materials", "has", "crystal_systems"),
                ],
                "rev_edge_types": [
                    ("elements", "rev_has", "materials"),
                    ("space_groups", "rev_has", "materials"),
                    ("crystal_systems", "rev_has", "materials"),
                ],
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
                "data", "training_runs", "heterograph_encoder_general"
            ),
            "learning_rate": 0.001,
            "num_epochs": 20001,
            "eval_interval": 1000,
            "scheduler_milestones": [4000, 20000],
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
# print(mdb)


material_store = mdb.material_store


df = material_store.read(
    columns=[
        "elasticity.g_vrh",
        "elasticity.k_vrh",
        "core.volume",
        "core.density",
        "core.density_atomic",
        "core.nelements",
        "core.nsites",
    ],
    filters=[
        pc.field("elasticity.g_vrh") > 0,
        pc.field("elasticity.g_vrh") < 400,
    ],
).to_pandas()


print("-" * 100)
print(f"Max memory allocated: {torch.cuda.max_memory_allocated()}")

# ####################################################################################################
# # Model
# ####################################################################################################


# y_index = parent_data['materials'].y_index
z = df[
    [
        "core.volume",
        "core.density",
        "core.density_atomic",
        "core.nelements",
        # "core.nsites"
    ]
]
y = df["elasticity.g_vrh"]

z = torch.tensor(z.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32)

perm = torch.randperm(z.size(0))
train_perm = perm[: int(z.size(0) * CONFIG.data.train_ratio)]
test_perm = perm[int(z.size(0) * CONFIG.data.train_ratio) :]
print(f"N train: {len(train_perm)}, N test: {len(test_perm)}")

reg = linear_model.LinearRegression()
reg.fit(z[train_perm].cpu().numpy(), y[train_perm].cpu().numpy())
y_pred = reg.predict(z[test_perm].cpu().numpy())
y_real = y[test_perm].cpu().numpy()

# y_pred = np.array([10**value for value in y_pred])
# y_real = np.array([10**value for value in y_real])
rmse = np.sqrt(np.mean((y_pred - y_real) ** 2))
mae = np.mean(np.abs(y_pred - y_real))
tmp_str = f"RMSE: {rmse:.4f}, MAE: {mae:.4f}|"
print(tmp_str)
