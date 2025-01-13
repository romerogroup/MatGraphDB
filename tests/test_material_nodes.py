import os
import shutil
import tempfile

import numpy as np
import pytest

from matgraphdb.materials.nodes.materials import MaterialStore


@pytest.fixture
def material_store():
    """Fixture that creates a temporary MaterialStore instance."""
    temp_dir = tempfile.mkdtemp()
    store = MaterialStore(storage_path=temp_dir)
    yield store
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_material():
    """Fixture that returns a sample material dictionary."""
    coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    species = ["Fe", "O"]
    lattice = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return dict(
        coords=coords,
        species=species,
        lattice=lattice,
        properties={"electronic_structure": {"band_gap": 1.0}},
    )


def test_create_material(material_store, sample_material):
    """Test creating a single material."""
    material_store.create_material(**sample_material)

    table = material_store.read()

    assert "formula" in table.column_names
    assert "electronic_structure.band_gap" in table.column_names
    assert table["electronic_structure.band_gap"].combine_chunks().to_pylist()[0] == 1.0

    # Test pandas conversion
    df = table.to_pandas()
    assert df.shape[0] == 1
    assert "formula" in df.columns
    assert "electronic_structure.band_gap" in df.columns
    assert df.iloc[0]["electronic_structure.band_gap"] == 1.0


def test_create_materials(material_store, sample_material):
    """Test creating multiple materials at once."""
    materials = [sample_material for _ in range(10)]

    material_store.create_materials(materials)
    table = material_store.read()
    df = table.to_pandas()

    assert df.shape[0] == 10
    assert "electronic_structure.band_gap" in df.columns


def test_update_material(material_store, sample_material):
    """Test updating an existing material."""
    # Create initial material
    material_store.create_material(**sample_material)

    table = material_store.read()
    df = table.to_pandas()
    assert df.shape[0] == 1
    assert "electronic_structure.band_gap" in df.columns
    assert df.iloc[0]["electronic_structure.band_gap"] == 1.0

    # Update the material
    update_dict = [
        {
            "id": 0,
            "electronic_structure": {"band_gap": 2.0},
            "lattice": [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        }
    ]
    material_store.update_materials(update_dict)

    # Verify update
    table = material_store.read()
    df = table.to_pandas()

    assert df.iloc[0]["electronic_structure.band_gap"] == 2.0
    assert np.array_equal(
        table["lattice"].combine_chunks().to_numpy_ndarray()[0],
        np.array(update_dict[0]["lattice"]),
    )


def test_delete_materials(material_store, sample_material):
    """Test deleting materials."""
    # Create materials
    materials = [sample_material for _ in range(10)]
    material_store.create_materials(materials)

    # Delete one material
    material_store.delete_materials(ids=[0])

    # Verify deletion
    table = material_store.read()
    df = table.to_pandas()
    assert df.shape[0] == 9
