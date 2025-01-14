import os
import shutil

import pandas as pd
import pyarrow as pa
import pytest
import torch
from torch_geometric.data import HeteroData

from matgraphdb.core.graph_db import GraphDB
from matgraphdb.pyg.builder import GraphBuilder


@pytest.fixture
def tmp_dir(tmp_path):
    """Fixture for temporary directory."""
    tmp_dir = str(tmp_path)
    yield tmp_dir
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


@pytest.fixture
def graph_db(tmp_dir):
    """Fixture to create a GraphDB instance with test data."""
    db = GraphDB(storage_path=tmp_dir)

    # Add test materials
    materials_data = pd.DataFrame(
        {
            "core.volume": [100.0, 200.0, 300.0],
            "core.density": [1.0, 2.0, 3.0],
            "elasticity.g_voigt": [10.0, 20.0, 30.0],
        }
    )
    db.add_nodes("materials", materials_data)

    # Add test crystal systems
    crystal_systems = pd.DataFrame({"name": ["cubic", "hexagonal"]})
    db.add_nodes("crystal_systems", crystal_systems)

    # Add test edges
    edges_data = pd.DataFrame(
        {
            "source_id": [0, 1, 2],
            "target_id": [0, 0, 1],
            "source_type": ["materials"] * 3,
            "target_type": ["crystal_systems"] * 3,
            "edge_type": ["material_crystalSystem_has"] * 3,
            "weight": [0.5, 0.7, 0.9],
        }
    )
    db.add_edges("material_crystalSystem_has", edges_data)

    return db


@pytest.fixture
def graph_builder(graph_db):
    """Fixture to create a GraphBuilder instance."""
    return GraphBuilder(graph_db)


def test_init(graph_builder):
    """Test GraphBuilder initialization."""
    assert isinstance(graph_builder.hetero_data, HeteroData)
    assert graph_builder.graph_db is not None
    assert len(graph_builder.node_id_mappings) == 0


def test_add_node_type(graph_builder):
    """Test adding node type with features."""
    graph_builder.add_node_type("materials", columns=["core.volume", "core.density"])

    # Check node features were added correctly
    assert "materials" in graph_builder.node_types
    assert graph_builder.hetero_data["materials"].x.shape == (3, 2)
    assert graph_builder.hetero_data["materials"].num_nodes == 3


def test_add_target_node_property(graph_builder):
    """Test adding target properties to nodes."""
    # First add the node type
    graph_builder.add_node_type("materials", columns=["core.volume", "core.density"])

    # Then add target property
    graph_builder.add_target_node_property("materials", columns=["elasticity.g_voigt"])

    # Check target properties were added correctly
    assert graph_builder.hetero_data["materials"].y is not None
    assert graph_builder.hetero_data["materials"].y.shape == (3, 1)
    assert graph_builder.hetero_data["materials"].out_channels == 1


def test_add_edge_type(graph_builder):
    """Test adding edge type with features."""
    # First add the node types
    graph_builder.add_node_type("materials")
    graph_builder.add_node_type("crystal_systems")

    # Then add edge type
    graph_builder.add_edge_type("material_crystalSystem_has", columns=["weight"])

    # Check edge features were added correctly
    edge_type = ("materials", "material_crystalSystem_has", "crystal_systems")
    assert edge_type in graph_builder.edge_types
    assert graph_builder.hetero_data[edge_type].edge_attr is not None
    assert graph_builder.hetero_data[edge_type].edge_attr.shape == (3, 1)


def test_add_target_edge_property(graph_builder):
    """Test adding target properties to edges."""
    # First add nodes and edges
    graph_builder.add_node_type("materials")
    graph_builder.add_node_type("crystal_systems")
    graph_builder.add_edge_type("material_crystalSystem_has")

    # Then add target property
    graph_builder.add_target_edge_property(
        "material_crystalSystem_has", columns=["weight"]
    )

    # Check target properties were added correctly
    edge_type = ("materials", "material_crystalSystem_has", "crystal_systems")
    assert graph_builder.hetero_data[edge_type].y is not None
    assert graph_builder.hetero_data[edge_type].y.shape == (3, 1)
    assert graph_builder.hetero_data[edge_type].out_channels == 1


def test_save_load(graph_builder, tmp_dir, graph_db):
    """Test saving and loading the graph."""
    # Build a test graph
    graph_builder.add_node_type("materials", columns=["core.volume", "core.density"])
    graph_builder.add_node_type("crystal_systems")
    graph_builder.add_edge_type("material_crystalSystem_has", columns=["weight"])

    # Save the graph
    save_path = os.path.join(tmp_dir, "test_graph.pt")
    graph_builder.save(save_path)

    # Load the graph
    loaded_builder = GraphBuilder.load(graph_db, save_path)

    # Check if loaded graph matches original
    assert len(loaded_builder.node_types) == len(graph_builder.node_types)
    assert len(loaded_builder.edge_types) == len(graph_builder.edge_types)

    # Check if node features match
    assert torch.equal(
        loaded_builder.hetero_data["materials"].x,
        graph_builder.hetero_data["materials"].x,
    )


def test_invalid_node_type(graph_builder):
    """Test adding invalid node type."""
    with pytest.raises(Exception):
        graph_builder.add_node_type("invalid_type")


def test_invalid_edge_type(graph_builder):
    """Test adding edge type with missing node types."""
    with pytest.raises(ValueError):
        graph_builder.add_edge_type("material_crystalSystem_has")
