import pytest
import pandas as pd
import pyarrow as pa
import os
import shutil
from matgraphdb.core.edges import EdgeStore

@pytest.fixture
def temp_storage(tmp_path):
    """Fixture to create and cleanup a temporary storage directory"""
    storage_dir = tmp_path / "test_edge_store"
    yield str(storage_dir)
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

@pytest.fixture
def edge_store(temp_storage):
    """Fixture to create an EdgeStore instance"""
    return EdgeStore(temp_storage)

@pytest.fixture
def sample_edge_data():
    """Fixture providing sample edge data with required fields"""
    return {
        'source_id': [1, 2],
        'target_id': [3, 4],
        'source_type': ['node_a', 'node_a'],
        'target_type': ['node_b', 'node_b'],
        'weight': [0.5, 0.7]
    }

def test_edge_store_initialization(temp_storage):
    """Test that EdgeStore initializes correctly and creates the storage directory"""
    store = EdgeStore(temp_storage)
    assert os.path.exists(temp_storage)
    assert store.db is not None

def test_create_edges_from_dict(edge_store, sample_edge_data):
    """Test creating edges from a dictionary"""
    edge_store.create_edges(sample_edge_data)
    
    # Read back and verify
    result_table = edge_store.read_edges()
    result_df = result_table.to_pandas()
    assert len(result_df) == 2
    assert all(field in result_df.columns for field in EdgeStore.required_fields)
    assert list(result_df['source_id']) == [1, 2]
    assert list(result_df['target_id']) == [3, 4]

def test_create_edges_missing_required_fields(edge_store):
    """Test that creating edges with missing required fields raises an error"""
    invalid_data = {
        'source_id': [1, 2],
        'target_id': [3, 4],
        # Missing source_type and target_type
        'weight': [0.5, 0.7]
    }
    
    with pytest.raises(ValueError, match="Edge data is missing required fields"):
        edge_store.create_edges(invalid_data)

def test_create_edges_from_dataframe(edge_store, sample_edge_data):
    """Test creating edges from a pandas DataFrame"""
    df = pd.DataFrame(sample_edge_data)
    edge_store.create_edges(df)
    
    result_table = edge_store.read_edges()
    result_df = result_table.to_pandas()
    assert len(result_df) == 2
    assert all(result_df['source_id'] == df['source_id'])
    assert all(result_df['target_id'] == df['target_id'])

def test_read_edges_with_filters(edge_store, sample_edge_data):
    """Test reading edges with specific filters"""
    edge_store.create_edges(sample_edge_data)
    
    # Read with column filter
    result_table = edge_store.read_edges(columns=['id', 'source_id', 'target_id'])
    result_df = result_table.to_pandas()
    assert list(result_df.columns) == ['id', 'source_id', 'target_id']
    
    # Read with ID filter
    first_result_table = edge_store.read_edges()
    first_result_df = first_result_table.to_pandas()
    first_id = first_result_df['id'].iloc[0]
    filtered_result_table = edge_store.read_edges(ids=[first_id])
    filtered_result_df = filtered_result_table.to_pandas()
    
    assert len(filtered_result_df) == 1
    assert filtered_result_df['id'].iloc[0] == first_id

def test_update_edges(edge_store, sample_edge_data):
    """Test updating existing edges"""
    edge_store.create_edges(sample_edge_data)
    
    # Get the IDs
    existing_edges_table = edge_store.read_edges()
    existing_edges_df = existing_edges_table.to_pandas()
    first_id = existing_edges_df['id'].iloc[0]
    
    # Update the first edge
    update_data = {
        'id': [first_id],
        'source_id': [1],
        'target_id': [5],
        'source_type': ['node_a'],
        'target_type': ['node_b'],
        'weight': [0.9]
    }
    edge_store.update_edges(update_data)
    
    # Verify update
    updated_edges_table = edge_store.read_edges()
    updated_edges_df = updated_edges_table.to_pandas()
    assert updated_edges_df[updated_edges_df['id'] == first_id]['weight'].iloc[0] == 0.9
    assert updated_edges_df[updated_edges_df['id'] == first_id]['target_id'].iloc[0] == 5

def test_delete_edges(edge_store, sample_edge_data):
    """Test deleting edges"""
    edge_store.create_edges(sample_edge_data)
    
    # Get the IDs
    existing_edges_table = edge_store.read_edges()
    existing_edges_df = existing_edges_table.to_pandas()
    first_id = existing_edges_df['id'].iloc[0]
    
    # Delete one edge
    edge_store.delete_edges(ids=[first_id])
    
    # Verify deletion
    remaining_edges_table = edge_store.read_edges()
    remaining_edges_df = remaining_edges_table.to_pandas()
    assert len(remaining_edges_df) == 1
    assert first_id not in remaining_edges_df['id'].values

def test_delete_columns(edge_store, sample_edge_data):
    """Test deleting specific columns"""
    edge_store.create_edges(sample_edge_data)
    
    # Delete the weight column
    edge_store.delete_edges(columns=['weight'])
    
    # Verify column deletion
    result_table = edge_store.read_edges()
    result_df = result_table.to_pandas()
    assert 'weight' not in result_df.columns
    assert all(field in result_df.columns for field in EdgeStore.required_fields)

def test_create_edges_with_schema(edge_store, sample_edge_data):
    """Test creating edges with a specific schema"""
    schema = pa.schema([
        ('source_id', pa.int64()),
        ('target_id', pa.int64()),
        ('source_type', pa.string()),
        ('target_type', pa.string()),
        ('weight', pa.float64())
    ])
    
    edge_store.create_edges(sample_edge_data, schema=schema)
    result_table = edge_store.read_edges()
    result_df = result_table.to_pandas()
    assert len(result_df) == 2
    assert result_df['weight'].dtype == 'float64'

def test_normalize_edges(edge_store, sample_edge_data):
    """Test the normalize operation"""
    edge_store.create_edges(sample_edge_data)
    
    # This should not raise any errors
    edge_store.normalize_edges()
    
    # Verify data is still accessible after normalization
    result_table = edge_store.read_edges()
    result_df = result_table.to_pandas()
    assert len(result_df) == 2
