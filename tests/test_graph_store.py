import pytest
import pandas as pd
import pyarrow as pa
import os
import shutil
from matgraphdb.core.graphs import GraphStore

@pytest.fixture
def temp_storage(tmp_path):
    """Fixture to create and cleanup a temporary storage directory"""
    storage_dir = tmp_path / "test_graph"
    yield str(storage_dir)
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

@pytest.fixture
def graph(temp_storage):
    """Fixture to create a GraphStore instance"""
    return GraphStore(temp_storage)

@pytest.fixture
def sample_nodes():
    """Fixture providing sample node data"""
    return {
        'user': {
            'name': ['Alice', 'Bob'],
            'age': [25, 30]
        },
        'item': {
            'name': ['Item1', 'Item2'],
            'price': [10.0, 20.0]
        }
    }

@pytest.fixture
def sample_edges():
    """Fixture providing sample edge data"""
    return {
        'source_id': [0, 1],
        'target_id': [0, 1],
        'source_type': ['user', 'user'],
        'target_type': ['item', 'item'],
        'weight': [0.5, 0.7]
    }

def test_graph_initialization(temp_storage):
    """Test that Graph initializes correctly and creates required directories"""
    graph = GraphStore(temp_storage)
    assert os.path.exists(os.path.join(temp_storage, 'nodes'))
    assert os.path.exists(os.path.join(temp_storage, 'edges'))
    assert isinstance(graph.node_stores, dict)
    assert isinstance(graph.edge_stores, dict)

def test_add_node_type(graph):
    """Test adding a new node type"""
    node_store = graph.add_node_type('user')
    assert 'user' in graph.node_stores
    assert node_store == graph.node_stores['user']
    
    # Test getting existing node store
    same_store = graph.add_node_type('user')
    assert same_store == node_store

def test_add_edge_type(graph):
    """Test adding a new edge type"""
    edge_store = graph.add_edge_type('likes')
    assert 'likes' in graph.edge_stores
    assert edge_store == graph.edge_stores['likes']
    
    # Test getting existing edge store
    same_store = graph.add_edge_type('likes')
    assert same_store == edge_store

def test_create_and_read_nodes(graph, sample_nodes):
    """Test creating and reading nodes"""
    # Create nodes
    graph.create_nodes('user', sample_nodes['user'])
    graph.create_nodes('item', sample_nodes['item'])
    
    # Read and verify user nodes
    user_table = graph.read_nodes('user')
    user_df = user_table.to_pandas()
    assert len(user_df) == 2
    assert list(user_df['name']) == ['Alice', 'Bob']
    
    # Read and verify item nodes
    item_table = graph.read_nodes('item')
    item_df = item_table.to_pandas()
    assert len(item_df) == 2
    assert list(item_df['name']) == ['Item1', 'Item2']

def test_create_and_read_edges(graph, sample_nodes, sample_edges):
    """Test creating and reading edges"""
    # First create the nodes that the edges will reference
    graph.create_nodes('user', sample_nodes['user'])
    graph.create_nodes('item', sample_nodes['item'])
    
    # Create edges
    graph.create_edges('likes', sample_edges)
    
    # Read and verify edges
    edge_table = graph.read_edges('likes')
    edge_df = edge_table.to_pandas()
    assert len(edge_df) == 2
    assert list(edge_df['source_id']) == [0, 1]
    assert list(edge_df['target_id']) == [0, 1]

def test_update_nodes(graph, sample_nodes):
    """Test updating nodes"""
    graph.create_nodes('user', sample_nodes['user'])
    
    # Update a node
    update_data = {
        'id': [1],
        'name': ['Alice Updated'],
        'age': [26]
    }
    graph.update_nodes('user', update_data)
    
    # Verify update
    user_table = graph.read_nodes('user')
    user_df = user_table.to_pandas()
    assert user_df[user_df['id'] == 1]['name'].iloc[0] == 'Alice Updated'
    assert user_df[user_df['id'] == 1]['age'].iloc[0] == 26

def test_update_edges(graph, sample_nodes, sample_edges):
    """Test updating edges"""
    graph.create_nodes('user', sample_nodes['user'])
    graph.create_nodes('item', sample_nodes['item'])
    graph.create_edges('likes', sample_edges)
    
    # Update an edge
    update_data = {
        'id': [0],  # Assuming first edge has id 0
        'source_id': [0],
        'source_type': ['user'],
        'target_id': [1],
        'target_type': ['item'],
        'weight': [0.9]
    }
    graph.update_edges('likes', update_data)
    
    # Verify update
    edge_table = graph.read_edges('likes')
    edge_df = edge_table.to_pandas()
    assert edge_df[edge_df['id'] == 0]['weight'].iloc[0] == 0.9

def test_delete_nodes(graph, sample_nodes):
    """Test deleting nodes"""
    graph.create_nodes('user', sample_nodes['user'])
    
    # Get initial node count
    initial_table = graph.read_nodes('user')
    initial_count = len(initial_table)
    
    # Delete a node
    graph.delete_nodes('user', ids=[1])
    
    # Verify deletion
    final_table = graph.read_nodes('user')
    final_df = final_table.to_pandas()
    assert len(final_df) == initial_count - 1
    assert 1 not in final_df['id'].values

def test_delete_edges(graph, sample_nodes, sample_edges):
    """Test deleting edges"""
    graph.create_nodes('user', sample_nodes['user'])
    graph.create_nodes('item', sample_nodes['item'])
    graph.create_edges('likes', sample_edges)
    
    # Get initial edge count
    initial_table = graph.read_edges('likes')
    initial_count = len(initial_table)
    
    # Delete an edge
    first_id = initial_table.to_pandas()['id'].iloc[0]
    graph.delete_edges('likes', ids=[first_id])
    
    # Verify deletion
    final_table = graph.read_edges('likes')
    final_df = final_table.to_pandas()
    assert len(final_df) == initial_count - 1
    assert first_id not in final_df['id'].values

def test_edge_reference_validation(graph, sample_nodes):
    """Test that edge creation validates node references"""
    # Create only user nodes, not item nodes
    graph.create_nodes('user', sample_nodes['user'])
    
    # Try to create edges with invalid target nodes
    invalid_edges = {
        'source_id': [0],
        'target_id': [999],  # Non-existent target ID
        'source_type': ['user'],
        'target_type': ['item'],
        'weight': [0.5]
    }
    
    # Should raise ValueError due to missing target nodes
    with pytest.raises(ValueError, match="No node store found for target_node_type='item'"):
        graph.create_edges('likes', invalid_edges)

def test_normalize_operations(graph, sample_nodes, sample_edges):
    """Test normalize operations for both nodes and edges"""
    graph.create_nodes('user', sample_nodes['user'])
    graph.create_nodes('item', sample_nodes['item'])
    graph.create_edges('likes', sample_edges)
    
    # These should not raise any errors
    graph.normalize_nodes('user')
    graph.normalize_nodes('item')
    graph.normalize_edges('likes')
    
    # Verify data is still accessible
    user_table = graph.read_nodes('user')
    item_table = graph.read_nodes('item')
    edge_table = graph.read_edges('likes')
    
    assert len(user_table) > 0
    assert len(item_table) > 0
    assert len(edge_table) > 0
