import pytest
import os
import shutil
import pandas as pd
import pyarrow as pa

from matgraphdb.stores.graph_db import GraphDB
from matgraphdb.stores.nodes import ElementNodes
from matgraphdb.materials.edges import element_element_neighborsByGroupPeriod

@pytest.fixture
def tmp_dir(tmp_path):
    """Fixture for temporary directory."""
    tmp_dir = str(tmp_path)
    yield tmp_dir
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

@pytest.fixture
def graphdb(tmp_dir):
    """Fixture to create a GraphDB instance."""
    return GraphDB(storage_path=tmp_dir)

@pytest.fixture
def element_store(tmp_dir):
    """Fixture to create an ElementNodes store with some test data."""
    store = ElementNodes(storage_path=os.path.join(tmp_dir, 'elements'))
    test_data = [
        {"atomic_number": 1, "symbol": "H", "extended_group": 1, "period": 1},
        {"atomic_number": 2, "symbol": "He", "extended_group": 18, "period": 1},
        {"atomic_number": 3, "symbol": "Li", "extended_group": 1, "period": 2},
        {"atomic_number": 4, "symbol": "Be", "extended_group": 2, "period": 2}
    ]
    store.create_nodes(test_data)
    return store

def test_add_edge_generator(graphdb, element_store):
    """Test adding an edge generator to the GraphDB."""
    generator_name = 'element_element_neighborsByGroupPeriod'
    
    # Add the generator
    graphdb.add_edge_generator(
        generator_name, 
        element_element_neighborsByGroupPeriod,
        generator_args={'element_store': element_store},
        generator_kwargs={}
    )
    
    # Verify the generator was added
    assert generator_name in graphdb.edge_generators
    
def test_run_edge_generator(graphdb, element_store):
    """Test running an edge generator and verify its output."""
    generator_name = 'element_element_neighborsByGroupPeriod'
    
    # Add and run the generator
    graphdb.add_edge_generator(
        generator_name, 
        element_element_neighborsByGroupPeriod,
        generator_args={'element_store': element_store},
        generator_kwargs={}
    )
    
    table = graphdb.run_edge_generator(generator_name)
    
    # Verify the output table has the expected structure
    assert isinstance(table, pa.Table)
    expected_columns = {
        'source_id', 'target_id', 'source_type', 'target_type', 
        'weight', 'source_name', 'target_name', 'name',
        'source_extended_group', 'source_period',
        'target_extended_group', 'target_period'
    }
    assert set(table.column_names) == expected_columns
    
    # Convert to pandas for easier verification
    df = table.to_pandas()
    
    # Basic validation checks
    assert not df.empty, "Generator produced no edges"
    assert all(df['source_type'] == 'elements'), "Incorrect source_type"
    assert all(df['target_type'] == 'elements'), "Incorrect target_type"
    assert all(df['weight'] == 1.0), "Incorrect weight values"
    
    # Verify edge names are properly formatted
    assert all(df['name'].str.contains('_neighborsByGroupPeriod_')), "Edge names not properly formatted"

def test_edge_generator_persistence(tmp_dir, element_store):
    """Test that edge generators persist when reloading the GraphDB."""
    generator_name = 'element_element_neighborsByGroupPeriod'
    
    # Create initial graph instance and add generator
    graph = GraphDB(storage_path=tmp_dir)
    graph.add_edge_generator(
        generator_name, 
        element_element_neighborsByGroupPeriod,
        generator_args={'element_store': element_store},
        generator_kwargs={}
    )
    
    # Create new graph instance (simulating program restart)
    new_graph = GraphDB(storage_path=tmp_dir)
    
    # Verify generator was loaded
    assert generator_name in new_graph.edge_generators
    
    # Verify generator still works
    table = new_graph.run_edge_generator(generator_name)
    assert isinstance(table, pa.Table)
    assert not table.empty

def test_invalid_generator_args(graphdb):
    """Test that invalid generator arguments raise appropriate errors."""
    generator_name = 'element_element_neighborsByGroupPeriod'
    
    # Test missing required argument
    with pytest.raises(TypeError):
        graphdb.add_edge_generator(
            generator_name, 
            element_element_neighborsByGroupPeriod,
            generator_args={},  # Missing element_store
            generator_kwargs={}
        )
    
    # Test invalid element_store argument
    with pytest.raises(Exception):
        graphdb.add_edge_generator(
            generator_name, 
            element_element_neighborsByGroupPeriod,
            generator_args={'element_store': 'invalid_store'},
            generator_kwargs={}
        ) 