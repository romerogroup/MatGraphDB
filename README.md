# MatGraphDB

[Documentation][docs] | [PyPI][pypi] | [GitHub][github]

**MatGraphDB** is a Python package designed to simplify graph-based data management and analysis in materials and molecular science. It is built on top of `ParquetGraphDB` [ParquetDB][parquetdb], which is a graph database which uses Apache Parquet for storage. It enables researchers to efficiently transform complex theoretical data into structured graph representations, leveraging:

- **High-performance storage:** Utilizes Apache Parquet for scalable and rapid data access.
- **Automated workflows:** Converts theoretical and computational data into graph structures.
- **Robust data operations:** Offers comprehensive CRUD functionality and custom generators to maintain consistent relationships between entities.

## Table of Contents
- [MatGraphDB](#matgraphdb)
    - [Table of Contents](#table-of-contents)
    - [Installing](#installing)
    - [Usage](#usage)
    - [Contributing](#contributing)
    - [License](#license)
    - [Authors](#authors)


## Installing

### Regular install

#### Install via pip    

```bash
pip install matgraphdb
```


#### Install from github

```bash
git clone https://github.com/romerogroup/MatGraphDB.git
cd MatGraphDB
pip install -e .
```


### Install with ML dependencies

You may want to install the package with its ML dependencies. This will install the latest version of PyTorch and the PyTorch Geometric package. This will be dependent on the CUDA version you have installed. 

#### Easy install (cpu)

The easiest way to install the package with ML dependencies is to use the `[ml]` flag. 
```bash
pip install matgraphdb[ml]
```

#### Manual install (gpu)

Here is an example of how to install the package with GPU support with CUDA 11.8. If you have a different version of CUDA installed, you can replace the version numbers `cu118` with the appropriate version for your system. 


```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

pip install torch_geometric

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu118.html
```


## Usage

### Interacting with the materials database.

#### Initialize MatGraphDB
```python
from matgraphdb import MatGraphDB

# Initialize MatGraphDB
mgdb = MatGraphDB(storage_path="MatGraphDB")
print(mgdb)
```

#### Adding material properties

You can add any material to the database by either providing a `structure` or `coords`, `species`, and `lattice`, then using the `create_material` or `create_materials` function. 

Any material add to the database gets indexed. This is stored in the `id` column.

```python
from pymatgen.core import Structure

# Add material to the database
material_data_1 = {
    "structure": structure,
    "properties": {
        "material_id": "mp-1",
        "source": "example",
        "thermal_conductivity": {"value": 2.5, "unit": "W/mK"},
    },
}

# or by coords, species, lattice
material_data_2 = {
    "coords":  [[0, 0, 0], [0.5, 0.5, 0.5]],
    "species": ["Mg", "O"],
    "lattice": [[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
    "properties": {
        "material_id": "mp-2",
        "source": "example_manual",
        "band_gap": {"value": 1.2, "unit": "eV"},
    },
}

result = mgdb.create_material(
    coords=material_data_2["coords"],
    species=material_data_2["species"],
    lattice=material_data_2["lattice"],
    properties=material_data_2["properties"],
)
# Add material by structure
db.create_material(
                structure=material_data_1["structure"],
                properties=material_data_1["properties"])


materials=[material_data_1,material_data_2]

# Add multiple materials
mgdb.create_materials(materials)

```

####  Reading Materials
 
To read materials from the database, you can use the `read_materials` function. This function takes in a `columns` parameter, which specifies the columns to read from the database. The `filters` parameter specifies the filters to apply to the database. This will only read the matched materials to memory.

```python
materials = mgdb.read_materials( 
    columns=["material_id", "elements", "band_gap.value"],
    filters=[pc.field("band_gap.value") == 1.2]
    )
```

####  Updating Materials

To update materials in the database, you can use the `update_materials` function. For updates you must provide the `id` of the material you want to update. You can also provide the `update_keys` parameter to specify the columns to update on as well, this is usefull if you import an existing dataset from another database.

```python
update_data = [
    {
        "id": 0,
        "band_gap": {"value": 3.6, "unit": "eV"},
    },
]

materials = mgdb.update_materials(update_data)


update_data = [
    {
        "material_id": "mp-1",
        "band_gap": {"value": 3.6, "unit": "eV"},
    },
]
materials = mgdb.update_materials(update_data, update_keys=["material_id"])
```

#### Deleting Materials

To delete materials from the database, you can use the `delete_materials` function. You can provide a list of `ids` to delete.

```python
materials = mgdb.delete_materials(ids=[0])
```


## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub. More information can be found in the [CONTRIBUTING][contributing] file.

## License

This project is licensed under the MIT License. See the [LICENSE][license] file for details.


## Authors
Logan Lang,
Aldo Romero,
Eduardo Hernandez,


---

*Note: This project is in its initial stages. Features and APIs are subject to change. Please refer to the latest documentation and release notes for updates.*



[docs]: https://matgraphdb.readthedocs.io/en/latest/
[pypi]: https://pypi.org/project/matgraphdb/
[github]: https://github.com/romerogroup/MatGraphDB
[contributing]: https://github.com/romerogroup/MatGraphDB/blob/main/CONTRIBUTING.md
[license]: https://github.com/romerogroup/MatGraphDB/blob/main/LICENSE
[parquetdb]: https://github.com/lllangWV/ParquetDB