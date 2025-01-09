import os
import shutil

import pyarrow as pa
import pytest

from matgraphdb.materials.core import MatGraphDB
from matgraphdb.materials.edges import *
from matgraphdb.materials.nodes import *
from matgraphdb.utils.config import DATA_DIR, PKG_DIR, config

config.logging_config.loggers.matgraphdb.level = "DEBUG"
config.apply()

current_dir = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(current_dir, "test_data")


@pytest.fixture
def tmp_dir(tmp_path):
    """Fixture for temporary directory."""
    tmp_dir = str(tmp_path)
    yield tmp_dir
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


@pytest.fixture
def matgraphdb(tmp_dir, material_store):
    """Fixture to create a MatGraphDB instance."""
    return MatGraphDB(storage_path=tmp_dir, materials_store=material_store)


@pytest.fixture
def empty_matgraphdb(tmp_dir):
    """Fixture to create a MatGraphDB instance."""
    return MatGraphDB(storage_path=tmp_dir)


@pytest.fixture
def empty_material_store(tmp_dir):
    materials_path = os.path.join(tmp_dir, "materials")
    return MaterialNodes(storage_path=materials_path)


@pytest.fixture
def material_store():
    materials_path = os.path.join(TEST_DATA_DIR, "materials")
    return MaterialNodes(storage_path=materials_path)


@pytest.fixture
def element_store(tmp_dir):
    return ElementNodes(storage_path=tmp_dir)


@pytest.fixture
def node_generator_data(matgraphdb):
    generators = [
        elements,
        chemenvs,
        crystal_systems,
        magnetic_states,
        oxidation_states,
        space_groups,
        wyckoffs,
        material_sites,
        material_lattices,
    ]
    generators_args = [
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {"material_store_path": matgraphdb.material_nodes.db_path},
        {"material_store_path": matgraphdb.material_nodes.db_path},
    ]
    return matgraphdb, generators, generators_args


@pytest.fixture
def edge_generator_data(node_generator_data):
    matgraphdb, node_generators_list, node_generators_args = node_generator_data

    for i, generator in enumerate(node_generators_list[:]):
        generator_name = generator.__name__

        matgraphdb.add_node_generator(
            generator_name=generator_name,
            generator_func=generator,
            generator_args=node_generators_args[i],
            run_immediately=True,
        )

    generators = [
        element_element_neighborsByGroupPeriod,
        element_oxiState_canOccur,
        material_chemenv_containsSite,
        material_crystalSystem_has,
        material_element_has,
        material_lattice_has,
        material_spg_has,
        element_chemenv_canOccur,
    ]

    generators_kwargs = [
        {"element_store_path": matgraphdb.get_node_store("elements").db_path},
        {
            "element_store_path": matgraphdb.get_node_store("elements").db_path,
            "oxiState_store_path": matgraphdb.get_node_store(
                "oxidation_states"
            ).db_path,
        },
        {
            "material_store_path": matgraphdb.get_node_store("materials").db_path,
            "chemenv_store_path": matgraphdb.get_node_store("chemenvs").db_path,
        },
        {
            "material_store_path": matgraphdb.get_node_store("materials").db_path,
            "crystal_system_store_path": matgraphdb.get_node_store(
                "crystal_systems"
            ).db_path,
        },
        {
            "material_store_path": matgraphdb.get_node_store("materials").db_path,
            "element_store_path": matgraphdb.get_node_store("elements").db_path,
        },
        {
            "material_store_path": matgraphdb.get_node_store("materials").db_path,
            "lattice_store_path": matgraphdb.get_node_store(
                "material_lattices"
            ).db_path,
        },
        {
            "material_store_path": matgraphdb.get_node_store("materials").db_path,
            "spg_store_path": matgraphdb.get_node_store("space_groups").db_path,
        },
        {
            "element_store_path": matgraphdb.get_node_store("elements").db_path,
            "chemenv_store_path": matgraphdb.get_node_store("chemenvs").db_path,
            "material_store_path": matgraphdb.get_node_store("materials").db_path,
        },
    ]
    return matgraphdb, generators, generators_kwargs


@pytest.fixture
def test_material_data():
    """Fixture providing test material data."""
    from pymatgen.core import Structure

    return [
        {
            "structure": Structure(
                lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
                species=["Mg", "O"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            ),
            "properties": {"material_id": "mp-1"},
        },
        {
            "structure": Structure(
                lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
                species=["Mg", "O"],
                coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            ),
            "properties": {"material_id": "mp-2"},
        },
    ]


def test_initialize_matgraphdb(empty_matgraphdb):
    """Test if MatGraphDB initializes correctly with materials store."""
    # Check if materials node store exists
    matgraphdb = empty_matgraphdb
    assert (
        "materials" in matgraphdb.node_stores
    ), f"Materials node store not found in node_stores: {matgraphdb.node_stores}"
    assert isinstance(
        matgraphdb.material_nodes, MaterialNodes
    ), f"MaterialNodes instance not found in matgraphdb: {matgraphdb.material_nodes}"

    # Check if materials directory was created
    materials_path = os.path.join(matgraphdb.nodes_path, "materials")
    assert os.path.exists(
        materials_path
    ), f"Materials directory not created: {materials_path}"


def test_create_materials(empty_matgraphdb, test_material_data):
    """Test creating materials in the database."""
    matgraphdb = empty_matgraphdb
    matgraphdb.create_materials(test_material_data)

    # Read back the materials and verify
    materials = matgraphdb.read_materials()
    assert isinstance(materials, pa.Table), f"Materials not created: {materials}"
    assert materials.num_rows == 2, f"Materials not created: {materials.num_rows}"

    # Convert to dict for easier comparison
    materials_dict = materials.to_pydict()
    assert "material_id" in materials_dict, f"Materials not created: {materials_dict}"
    assert materials_dict["material_id"] == [
        "mp-1",
        "mp-2",
    ], f"Materials not created: {materials_dict['material_id']}"


def test_read_materials_with_filters(empty_matgraphdb, test_material_data):
    """Test reading materials with column filters."""
    matgraphdb = empty_matgraphdb
    matgraphdb.create_materials(test_material_data)

    # Read specific columns
    columns = ["material_id", "elements"]
    materials = matgraphdb.read_materials(columns=columns)

    assert isinstance(materials, pa.Table), f"Materials not read: {materials}"
    assert set(materials.column_names) == set(
        columns
    ), f"Columns not read: {materials.column_names}"

    # Read specific IDs
    materials = matgraphdb.read_materials(ids=[0])
    assert materials.num_rows == 1, f"Materials not read: {materials.num_rows}"
    assert materials.to_pydict()["material_id"] == [
        "mp-1"
    ], f"Materials not read: {materials.to_pydict()['material_id']}"


def test_update_materials(empty_matgraphdb, test_material_data):
    """Test updating existing materials."""
    matgraphdb = empty_matgraphdb
    matgraphdb.create_materials(test_material_data)

    # Update data
    update_data = {
        "material_id": ["mp-1"],
        "nelements": [3],
        "elements": [["Fe", "O", "H"]],
    }

    matgraphdb.update_materials(update_data, update_keys=["material_id"])

    # Verify update
    materials = matgraphdb.read_materials(
        columns=["material_id", "nelements", "elements"]
    )
    materials_dict = materials.to_pydict()
    assert (
        materials_dict["nelements"][0] == 3
    ), f"Materials not updated: {materials_dict['nelements'][0]}"
    assert materials_dict["elements"][0] == [
        "Fe",
        "O",
        "H",
    ], f"Materials not updated: {materials_dict['elements'][0]}"


def test_delete_materials(empty_matgraphdb, test_material_data):
    """Test deleting materials."""
    matgraphdb = empty_matgraphdb
    matgraphdb.create_materials(test_material_data)

    # Delete first material
    matgraphdb.delete_materials(ids=[0])

    # Verify deletion
    materials = matgraphdb.read_materials()
    assert materials.num_rows == 1, f"Materials not deleted: {materials.num_rows}"
    assert materials.to_pydict()["material_id"] == [
        "mp-2"
    ], f"Materials not deleted: {materials.to_pydict()['material_id']}"


def test_persistence(tmp_dir, test_material_data):
    """Test that materials persist when recreating the MatGraphDB instance."""
    # Create initial graph instance and add materials
    db = MatGraphDB(storage_path=tmp_dir)
    db.create_materials(test_material_data)

    # Create new graph instance (simulating program restart)
    new_db = MatGraphDB(storage_path=tmp_dir)

    # Verify materials persisted
    materials = new_db.read_materials()
    assert materials.num_rows == 2, f"Materials not persisted: {materials.num_rows}"
    materials_dict = materials.to_pydict()
    assert materials_dict["material_id"] == [
        "mp-1",
        "mp-2",
    ], f"Materials not persisted: {materials_dict['material_id']}"


def test_add_node_generators(node_generator_data):
    """Test adding a node generator."""
    matgraphdb, node_generators_list, node_generators_args = node_generator_data

    material_store = matgraphdb.material_nodes
    table = material_store.read()
    for i, generator in enumerate(node_generators_list[:]):
        generator_name = generator.__name__

        matgraphdb.add_node_generator(
            generator_name=generator_name,
            generator_func=generator,
            generator_args=node_generators_args[i],
            run_immediately=True,
        )

    generator_names = [generator.__name__ for generator in node_generators_list]
    assert all(
        name in matgraphdb.node_stores for name in generator_names
    ), f"Node generators not found in node_stores: {generator_names}"


def test_moving_matgraphdb(tmp_dir, edge_generator_data):
    """Test adding an edge generator."""
    matgraphdb, edge_generators, edge_generators_kwargs = edge_generator_data

    current_dir = matgraphdb.storage_path

    parent_dir = os.path.dirname(current_dir)
    new_dir = os.path.join(parent_dir, "new_dir")
    shutil.move(current_dir, new_dir)

    matgraphdb = MatGraphDB(storage_path=new_dir)
    edge_generators = [
        element_element_neighborsByGroupPeriod,
        element_oxiState_canOccur,
        material_chemenv_containsSite,
        material_crystalSystem_has,
        material_element_has,
        material_lattice_has,
        material_spg_has,
        element_chemenv_canOccur,
    ]

    edge_generators_args = [
        {"element_store_path": matgraphdb.get_node_store("elements").db_path},
        {
            "element_store_path": matgraphdb.get_node_store("elements").db_path,
            "oxiState_store_path": matgraphdb.get_node_store(
                "oxidation_states"
            ).db_path,
        },
        {
            "material_store_path": matgraphdb.get_node_store("materials").db_path,
            "chemenv_store_path": matgraphdb.get_node_store("chemenvs").db_path,
        },
        {
            "material_store_path": matgraphdb.get_node_store("materials").db_path,
            "crystal_system_store_path": matgraphdb.get_node_store(
                "crystal_systems"
            ).db_path,
        },
        {
            "material_store_path": matgraphdb.get_node_store("materials").db_path,
            "element_store_path": matgraphdb.get_node_store("elements").db_path,
        },
        {
            "material_store_path": matgraphdb.get_node_store("materials").db_path,
            "lattice_store_path": matgraphdb.get_node_store(
                "material_lattices"
            ).db_path,
        },
        {
            "material_store_path": matgraphdb.get_node_store("materials").db_path,
            "spg_store_path": matgraphdb.get_node_store("space_groups").db_path,
        },
        {
            "element_store_path": matgraphdb.get_node_store("elements").db_path,
            "chemenv_store_path": matgraphdb.get_node_store("chemenvs").db_path,
            "material_store_path": matgraphdb.get_node_store("materials").db_path,
        },
    ]

    for i, generator in enumerate(edge_generators[:]):
        generator_name = generator.__name__
        matgraphdb.add_edge_generator(
            generator_name=generator_name,
            generator_func=generator,
            generator_args=edge_generators_args[i],
            # generator_kwargs=edge_generators_kwargs[i],
            run_immediately=True,
        )

    edge_generators_names = [generator.__name__ for generator in edge_generators]
    assert all(
        name in matgraphdb.edge_stores for name in edge_generators_names
    ), f"Edge generators not found in edge_stores: {edge_generators_names}"
