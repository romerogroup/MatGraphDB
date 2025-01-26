import os

import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch import optim
from torch_geometric import nn as pyg_nn

from matgraphdb.materials.datasets.mp_near_hull import MPNearHull
from matgraphdb.pyg.data import HeteroGraphBuilder
from matgraphdb.utils.config import DATA_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def drop_duplicates(table, keys):
    """
    Drops duplicate rows from a PyArrow Table based on four specified keys, keeping the first occurrence.

    Parameters:
    - table: pyarrow.Table
        The input table from which duplicates will be removed.
    - keys: list of str
        A list of four column names that determine the uniqueness of rows.

    Returns:
    - pyarrow.Table
        A new table with duplicates removed, keeping the first occurrence of each unique key combination.
    """
    keys.append("id")
    groupby_keys = list(set(keys).union(set(["id"])))
    # Add an index column to track the original row positions
    table_dup = table.group_by(groupby_keys).aggregate([])
    print(table_dup.shape)
    t2 = table_dup.group_by(keys).aggregate([("id", "min")]).column("id_min")

    new_table = pc.take(table, t2)

    return new_table


mdb = MPNearHull()
current_dir = os.path.dirname(os.path.abspath(__file__))

# with open(os.path.join(current_dir, "schema.txt"), "w", encoding="utf-8") as f:
#     f.write(str(mdb))

# edge_store = mdb.get_edge_store("element_element_neighborsByGroupPeriod")

# table = edge_store.read(
#     columns=["source_id", "source_type", "target_id", "target_type", "id"]
# )
# print(f"Table Shape: {table.shape}")
# new_table = drop_duplicates(
#     table, keys=["source_id", "source_type", "target_id", "target_type"]
# )
# print(f"New Table Shape: {new_table.shape}")
builder = HeteroGraphBuilder(mdb)

builder.add_node_type(
    "materials",
    columns=[
        "core.volume",
        "core.density",
        "core.density_atomic",
        "core.nelements",
        "core.nsites",
    ],
)
builder.add_node_type(
    "elements",
    columns=[
        "atomic_mass",
        "radius_covalent",
        "radius_vanderwaals",
        "heat_specific",
    ],
)

builder.add_edge_type("element_element_neighborsByGroupPeriod")
builder.add_edge_type(
    "material_element_has",
    #   columns=["weight"]
)

data = builder.hetero_data

print(data)
print(data.num_nodes)
import numpy as np
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit

n_materials = data["materials"].num_nodes

material_edge_index = torch.tensor(
    [np.arange(n_materials), np.arange(n_materials)], dtype=torch.int64
)
print(material_edge_index.shape)
data["materials", "connectsSelf", "materials"].edge_index = material_edge_index


flip_edge_index = torch.flip(data["materials", "has", "elements"].edge_index, dims=[0])
print(flip_edge_index[:, :10])
print(data["materials", "has", "elements"].edge_index[:, :10])
data["elements", "occursIn", "materials"].edge_index = torch.flip(
    data["materials", "has", "elements"].edge_index, dims=[0]
)


def split_data_on_node_type(
    data,
    node_type_to_split,
    train_proportion=0.8,
    test_proportion=0.1,
    val_proportion=0.1,
):
    assert train_proportion + test_proportion + val_proportion == 1.0
    for node_type in data.node_types:
        train_mask = torch.zeros(data[node_type].num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data[node_type].num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data[node_type].num_nodes, dtype=torch.bool)

        num_nodes_for_type = data[node_type].num_nodes
        if node_type == node_type_to_split:
            # Determine indices for training, testing, and validation
            indices = torch.randperm(num_nodes_for_type)

            num_train = int(train_proportion * num_nodes_for_type)
            num_val = int(test_proportion * num_nodes_for_type)
            num_test = num_nodes_for_type - num_train - num_val

            train_mask[indices[:num_train]] = True
            val_mask[indices[num_train : num_train + num_val]] = True
            test_mask[indices[num_train + num_val :]] = True
        else:
            train_mask[:num_nodes_for_type] = True

        data[node_type].train_mask = train_mask
        data[node_type].test_mask = test_mask
        data[node_type].val_mask = val_mask
    return data


num_edge_types = len(data.edge_types)
num_node_types = len(data.node_types)
data.num_edges
transform = RandomLinkSplit(
    num_val=0.1,  # 10% for validation
    num_test=0.1,  # 10% for test
    is_undirected=False,  # Set to True if the graph is undirected
    edge_types=[("materials", "has", "elements")],  # Specify edge types
    rev_edge_types=None,  # If there are reverse edges
)
train_data, val_data, test_data = transform(data)
print(train_data)
print(val_data)
print(test_data)


# transform = RandomNodeSplit(
#     # split="train_rest",
#     num_splits=1,
#     num_train_per_class=20,
#     num_val=500,
#     num_test=1000,
#     key="y",
# )
# data = transform(data)
# print(data)


##########################################################################################################################
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = pyg_nn.SAGEConv((-1, -1), hidden_channels)
        self.conv2 = pyg_nn.SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


# model = GNN(hidden_channels=64, out_channels=64)
# model = pyg_nn.to_hetero(model, data.metadata(), aggr="sum")
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# print(model(test_data.x_dict, test_data.edge_index_dict))
##########################################################################################################################

metapath = [
    ("materials", "has", "elements"),
    ("elements", "neighborsByGroupPeriod", "elements"),
    ("elements", "occursIn", "materials"),
]

model = pyg_nn.MetaPath2Vec(
    data.edge_index_dict,
    embedding_dim=128,
    metapath=metapath,
    walk_length=20,
    context_size=7,
    walks_per_node=5,
    num_negative_samples=5,
    sparse=True,
).to(device)

loader = model.loader(batch_size=128, shuffle=True, num_workers=6)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


def train(epoch, log_steps=100, eval_steps=2000):
    model.train()

    total_loss = 0
    for i, (pos_rw, neg_rw) in enumerate(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_steps == 0:
            print(
                f"Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, "
                f"Loss: {total_loss / log_steps:.4f}"
            )
            total_loss = 0

        if (i + 1) % eval_steps == 0:
            acc = test()
            print(
                f"Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, " f"Acc: {acc:.4f}"
            )


@torch.no_grad()
def test(train_ratio=0.1):
    model.eval()

    z = model("author", batch=data["author"].y_index.to(device))
    y = data["author"].y

    perm = torch.randperm(z.size(0))
    train_perm = perm[: int(z.size(0) * train_ratio)]
    test_perm = perm[int(z.size(0) * train_ratio) :]

    return model.test(
        z[train_perm], y[train_perm], z[test_perm], y[test_perm], max_iter=150
    )


for epoch in range(1, 6):
    train(epoch)
    acc = test()
    print(f"Epoch: {epoch}, Accuracy: {acc:.4f}")


# def train():
#     model.train()
#     total_loss = total_examples = 0
#     for head_index, rel_type, tail_index in loader:
#         optimizer.zero_grad()
#         loss = model.loss(head_index, rel_type, tail_index)
#         loss.backward()
#         optimizer.step()
#         total_loss += float(loss) * head_index.numel()
#         total_examples += head_index.numel()
#     return total_loss / total_examples


# @torch.no_grad()
# def test(data):
#     model.eval()
#     return model.test(
#         head_index=data.edge_index[0],
#         rel_type=data.edge_type,
#         tail_index=data.edge_index[1],
#         batch_size=20000,
#         k=10,
#     )


# for epoch in range(1, 501):
#     loss = train()
#     print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
#     if epoch % 25 == 0:
#         rank, hits = test(val_data)
#         print(
#             f"Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, "
#             f"Val Hits@10: {hits:.4f}"
#         )

# rank, hits_at_10 = test(test_data)
# print(f"Test Mean Rank: {rank:.2f}, Test Hits@10: {hits_at_10:.4f}")

# # model = pyg_nn.kge.TransE()
