import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from matgraphdb import MaterialStore, MatGraphDB, generators

current_dir = Path(__file__).parent
TEST_DATA_DIR = current_dir / "test_data"

VERBOSE = 1

TMP_DIR = Path(tempfile.mkdtemp())


@pytest.fixture
def matgraphdb(material_store):
    """Fixture to create a MatGraphDB instance."""
    matgraphdb_path = TMP_DIR / "MatGraphDB"
    if matgraphdb_path.exists():
        shutil.rmtree(matgraphdb_path)
    return MatGraphDB(
        storage_path=matgraphdb_path, materials_store=material_store, verbose=VERBOSE
    )


@pytest.fixture
def empty_matgraphdb():
    """Fixture to create a MatGraphDB instance."""
    matgraphdb_path = TMP_DIR / "MatGraphDB"
    if matgraphdb_path.exists():
        shutil.rmtree(matgraphdb_path)
    return MatGraphDB(storage_path=matgraphdb_path, verbose=VERBOSE)


@pytest.fixture
def empty_material_store():
    materials_path = TMP_DIR / "material"
    if materials_path.exists():
        shutil.rmtree(materials_path)
    return MaterialStore(storage_path=materials_path, verbose=VERBOSE)


@pytest.fixture
def material_store():
    materials_path = TEST_DATA_DIR / "material"
    return MaterialStore(storage_path=materials_path, verbose=VERBOSE)


@pytest.fixture
def node_generator_data(matgraphdb):
    node_generators = [
        {"generator_func": generators.element},
        {"generator_func": generators.chemenv},
        {"generator_func": generators.crystal_system},
        {"generator_func": generators.magnetic_state},
        {"generator_func": generators.oxidation_state},
        {"generator_func": generators.space_group},
        {"generator_func": generators.wyckoff},
        {
            "generator_func": generators.material_site,
            "generator_args": {"material_store": matgraphdb.node_stores["material"]},
        },
        {
            "generator_func": generators.material_lattice,
            "generator_args": {"material_store": matgraphdb.node_stores["material"]},
        },
    ]

    return matgraphdb, node_generators


@pytest.fixture
def edge_generator_data(node_generator_data):
    matgraphdb, node_generators_list = node_generator_data

    for i, generator in enumerate(node_generators_list[:]):

        matgraphdb.add_node_generator(
            generator_func=generator.get("generator_func"),
            generator_args=generator.get("generator_args", None),
            run_immediately=True,
        )

    edge_generators = [
        {
            "generator_func": generators.element_element_neighborsByGroupPeriod,
            "generator_args": {"element_store": matgraphdb.node_stores["element"]},
        },
        {
            "generator_func": generators.element_oxiState_canOccur,
            "generator_args": {
                "element_store": matgraphdb.node_stores["element"],
                "oxiState_store": matgraphdb.node_stores["oxidation_state"],
            },
        },
        {
            "generator_func": generators.material_chemenv_containsSite,
            "generator_args": {
                "material_store": matgraphdb.node_stores["material"],
                "chemenv_store": matgraphdb.node_stores["chemenv"],
            },
        },
        {
            "generator_func": generators.material_crystalSystem_has,
            "generator_args": {
                "material_store": matgraphdb.node_stores["material"],
                "crystal_system_store": matgraphdb.node_stores["crystal_system"],
            },
        },
        {
            "generator_func": generators.material_element_has,
            "generator_args": {
                "material_store": matgraphdb.node_stores["material"],
                "element_store": matgraphdb.node_stores["element"],
            },
        },
        {
            "generator_func": generators.material_lattice_has,
            "generator_args": {
                "material_store": matgraphdb.node_stores["material"],
                "lattice_store": matgraphdb.node_stores["material_lattice"],
            },
        },
        {
            "generator_func": generators.material_spg_has,
            "generator_args": {
                "material_store": matgraphdb.node_stores["material"],
                "spg_store": matgraphdb.node_stores["space_group"],
            },
        },
        {
            "generator_func": generators.element_chemenv_canOccur,
            "generator_args": {
                "element_store": matgraphdb.node_stores["element"],
                "chemenv_store": matgraphdb.node_stores["chemenv"],
                "material_store": matgraphdb.node_stores["material"],
            },
        },
    ]
    return matgraphdb, edge_generators


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
        "material" in matgraphdb.node_stores
    ), f"Materials node store not found in node_stores: {matgraphdb.node_stores}"
    assert isinstance(
        matgraphdb.material_store, MaterialStore
    ), f"MaterialStore instance not found in matgraphdb: {matgraphdb.material_store}"

    # Check if materials directory was created
    materials_path = os.path.join(matgraphdb.nodes_path, "material")
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

    print(matgraphdb.summary(show_column_names=True))
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


def test_persistence(test_material_data):
    """Test that materials persist when recreating the MatGraphDB instance."""
    # Create initial graph instance and add materials
    matgraphdb_path = TMP_DIR / "MatGraphDB"
    if matgraphdb_path.exists():
        shutil.rmtree(matgraphdb_path)

    db = MatGraphDB(storage_path=matgraphdb_path, verbose=VERBOSE)
    db.create_materials(test_material_data)

    # Create new graph instance (simulating program restart)
    new_db = MatGraphDB(storage_path=matgraphdb_path, verbose=VERBOSE)

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
    matgraphdb, node_generators_list = node_generator_data

    material_store = matgraphdb.material_store
    
    for generator in node_generators_list[:]:
        generator_func = generator.get("generator_func")
        generator_args = generator.get("generator_args", None)
        matgraphdb.add_node_generator(
            generator_func=generator_func,
            generator_args=generator_args,
        )

    generator_names = [
        generator["generator_func"].__name__ for generator in node_generators_list
    ]
    assert all(
        name in matgraphdb.node_stores for name in generator_names
    ), f"Node generators not found in node_stores: {generator_names}"


def test_moving_matgraphdb(edge_generator_data):
    """Test adding an edge generator."""
    matgraphdb, edge_generators = edge_generator_data

    current_dir = Path(matgraphdb.storage_path)

    parent_dir = current_dir.parent
    new_dir = parent_dir / "new_dir"
    shutil.move(current_dir, new_dir)

    matgraphdb = MatGraphDB(storage_path=new_dir)
    edge_generators = [
        {
            "generator_func": generators.element_element_neighborsByGroupPeriod,
            "generator_args": {"element_store": matgraphdb.node_stores["element"]},
        },
        {
            "generator_func": generators.element_oxiState_canOccur,
            "generator_args": {
                "element_store": matgraphdb.node_stores["element"],
                "oxiState_store": matgraphdb.node_stores["oxidation_state"],
            },
        },
        {
            "generator_func": generators.material_chemenv_containsSite,
            "generator_args": {
                "material_store": matgraphdb.node_stores["material"],
                "chemenv_store": matgraphdb.node_stores["chemenv"],
            },
        },
        {
            "generator_func": generators.material_crystalSystem_has,
            "generator_args": {
                "material_store": matgraphdb.node_stores["material"],
                "crystal_system_store": matgraphdb.node_stores["crystal_system"],
            },
        },
        {
            "generator_func": generators.material_element_has,
            "generator_args": {
                "material_store": matgraphdb.node_stores["material"],
                "element_store": matgraphdb.node_stores["element"],
            },
        },
        {
            "generator_func": generators.material_lattice_has,
            "generator_args": {
                "material_store": matgraphdb.node_stores["material"],
                "lattice_store": matgraphdb.node_stores["material_lattice"],
            },
        },
        {
            "generator_func": generators.material_spg_has,
            "generator_args": {
                "material_store": matgraphdb.node_stores["material"],
                "spg_store": matgraphdb.node_stores["space_group"],
            },
        },
        {
            "generator_func": generators.element_chemenv_canOccur,
            "generator_args": {
                "element_store": matgraphdb.node_stores["element"],
                "chemenv_store": matgraphdb.node_stores["chemenv"],
                "material_store": matgraphdb.node_stores["material"],
            },
        },
    ]

    for generator in edge_generators[:]:
        generator_func = generator.get("generator_func")
        generator_args = generator.get("generator_args", None)
        matgraphdb.add_edge_generator(
            generator_func=generator_func,
            generator_args=generator_args,
            run_immediately=True,
        )

    edge_generators_names = [
        generator["generator_func"].__name__ for generator in edge_generators
    ]
    assert all(
        name in matgraphdb.edge_stores for name in edge_generators_names
    ), f"Edge generators not found in edge_stores: {edge_generators_names}"


def test_dependency_updates(matgraphdb, node_generator_data):
    matgraphdb, node_generators_list = node_generator_data

    for generator in node_generators_list[:]:
        generator_func = generator.get("generator_func")
        generator_args = generator.get("generator_args", None)
        matgraphdb.add_node_generator(
            generator_func=generator_func,
            generator_args=generator_args,
            run_immediately=True,
        )

    edge_generators = [
        {
            "generator_func": generators.material_crystalSystem_has,
            "generator_args": {
                "material_store": matgraphdb.node_stores["material"],
                "crystal_system_store": matgraphdb.node_stores["crystal_system"],
            },
        },
    ]

    for generator in edge_generators[:]:
        generator_func = generator.get("generator_func")
        generator_args = generator.get("generator_args", None)
        matgraphdb.add_edge_generator(
            generator_func=generator_func,
            generator_args=generator_args,
            run_immediately=True,
        )

    data = pd.DataFrame(
        {
            "material_id": [1],
            "core.material_id": ["mp-1111111111"],
            "symmetry.crystal_system": ["Cubic"],
        }
    )

    matgraphdb.add_generator_dependency(
        generator_name="material_crystalSystem_has",
    )
    # Adding nodes
    matgraphdb.add_nodes(node_type="material", data=data)
    df = matgraphdb.read_nodes(
        "material",
        columns=["id", "symmetry.crystal_system"],
        filters=[pc.field("id") == 1000],
    ).to_pandas()
    assert df.shape[0] == 1
    assert df.iloc[0]["symmetry.crystal_system"] == "Cubic"

    df = matgraphdb.read_edges("material_crystalSystem_has").to_pandas()
    assert df.shape == (1001, 10)
    df = df[df["source_id"] == 1000]
    assert df.iloc[0]["target_id"] == 6  # Cubic id

    df = matgraphdb.read_nodes("material", columns=["core.material_id"]).to_pandas()

    # Updating nodes
    data = pd.DataFrame(
        {
            "id": [1000],
            "material_id": [1],
            "core.material_id": ["mp-1111111111"],
            "symmetry.crystal_system": ["Hexagonal"],
        }
    )
    matgraphdb.update_nodes("material", data)

    df = matgraphdb.read_nodes(
        "material",
        columns=["id", "symmetry.crystal_system"],
        filters=[pc.field("id") == 1000],
    ).to_pandas()
    assert df.shape[0] == 1
    assert df.iloc[0]["symmetry.crystal_system"] == "Hexagonal"

    df = matgraphdb.read_edges("material_crystalSystem_has").to_pandas()
    assert df.shape == (1001, 10)
    df = df[df["source_id"] == 1000]
    assert df.iloc[0]["target_id"] == 5  # Hexagonal id

    matgraphdb.delete_nodes("material", ids=[1000])

    df = matgraphdb.read_nodes(
        "material",
        columns=["id", "symmetry.crystal_system"],
        filters=[pc.field("id") == 1000],
    ).to_pandas()
    assert df.shape[0] == 0

    df = matgraphdb.read_edges("material_crystalSystem_has").to_pandas()
    assert df.shape == (1000, 10)
