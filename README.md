# MatGraphDB


## Introduction
MatGraphDB is a Python package designed to interface with primary and graph databases for the purpose of material analysis. The package facilitates complex computational tasks, including Density Functional Theory (DFT) calculations, through its modular components - `DataManager`, `CalcManager`, and `GraphManager`. This system is structured to handle vast datasets of materials data, encoding their properties and relationships within a graph-based analytical model. The ultimate goal of MatGraphDB is to leverage graph databases, specifically Neo4j, to enable advanced analysis and discovery in the realm of material science.

## System Architecture
![System Architecture of MatGraphDB](figures/package_database_interface.svg)
*Figure 1: The MatGraphDB architecture showcasing the interconnections between its components. `CalcManager` manages complex calculations, while `DataManager` oversees the interaction with JSON databases and encodings. The `GraphManager` facilitates operations with the Neo4j graph database, utilizing functions like `create_nodes()` and `create_relationships()` to represent materials and their interactions as graph elements stored in CSV format. The architecture illustrates the flow from the MatGraphDB package processing to the primary database, culminating in the Neo4j Graph Database for material analysis.*


## Getting Started

### Installing the data

You can install the data here:




### Setting up Conda environment
Navigate to the root directory. Then do the following


**Windows**
```bash
conda env create -f env_win.yml
```

**Linux**
```bash
conda env create -f env_linux.yml
```

To activeate the enviornment use:

```bash
conda activate matgraphdb
```

### Neo4j Interface

#### Bulk Importing Data
The best way to put large amounts of data into neo4j is to use the **neo4j-admin** to import csv files when the dbms is not runnning. This allows to put data into the database in a parallel way without having to worry abour ACID operations
To put nodes and relationships csv's produced into neo 4j follow these steps:

1. Make sure the dbms you want to put the database in is not running.
2. Open the terminal for the particular dbms.
3. run the following command
for neo4j==4.*
```bash
.\bin\neo4j-admin.bat import --database test --nodes import\elements.csv --relationships import\Element_Element.csv
```

for neo4j==5.*
```bash
.\bin\neo4j-admin.bat database import full --nodes import\elements.csv --relationships import\Element_Element.csv --overwrite-destination test
```
4. Start the dbms. Then create a new databse with same name as in the previous command. "test"


#### Creating vector index on an embedding of a material 

```cypher
CREATE VECTOR INDEX `material-MEGNET-embeddings`
FOR (n :Material) ON (n.`MEGNet-MP-2018`) 
OPTIONS {indexConfig:{
    `vector.dimensions`:160,
    `vector.similarity_function`:'cosine'}}
```

Check if vector indexing worked

```cypher
SHOW VECTOR INDEXES YIELD name, type, labelsOrTypes, properties, options
```

Example of how to call similar nodes for a given node

```bash
MATCH (m: Material {`material_id`:'mp-1000'})
CALL db.index.vector.queryNodes('material-MEGNET-embeddings', 10, m.`MEGNet-MP-2018`)
YIELD node as similarMaterial, score
RETURN m,similarMaterial, score
```

### Adjusting configs


## Usage
Interacting with the json database:

### PrimaryDatabase
**Checking properties**
```python
from matgraphdb import DatabaseManager

db=DatabaseManager()

success,failed=db.check_property(property_name="band_gap")

```

**Adding material properties**

```python
from matgraphdb import DatabaseManager

db=DatabaseManager()
structure = Structure(
        Lattice.cubic(3.0),
        ["C", "C"],  # Elements
        [
            [0, 0, 0],          # Coordinates for the first Si atom
            [0.25, 0.25, 0.25],  # Coordinates for the second Si atom (basis of the diamond structure)
        ]
    )

# Add material by structure
db.create_material(structure=structure)

# Add material by composition
db.create_material(structure="BaTe")
```

### Creating Graphs

```python
from matgraphdb.graph import create_nodes, create_relationships
from matgraphdb.graph.create_relationship_csv import create_material_element_task
from matgraphdb.graph.node_types import SPG_NAMES
from matgraphdb.utils import  NODE_DIR,RELATIONSHIP_DIR

save_path = os.path.join(NODE_DIR)
create_nodes(node_names=SPG_NAMES, 
            node_type='SpaceGroup', 
            node_prefix='spg', 
            filepath=os.path.join(save_path, 'spg.csv'))

# Create relationships between materials and elements
# mp_task is a function that will open a material json and define the relationships.
create_relationships(node_a_csv=os.path.join(NODE_DIR,'materials.csv'),
                    node_b_csv=os.path.join(NODE_DIR,'elements.csv'), 
                    mp_task=create_material_element_task,
                    connection_name='COMPOSED_OF',
                    filepath=os.path.join(save_path,f'materials_elements.csv'))
```


### Interacting with the Graph Databse

**List Database Schema**
```python
from matgraphdb import GraphDatabase

with GraphDatabase() as session:
    schema_list=session.list_schema()
```

**Execute Cypher Statement**
```python
from matgraphdb import GraphDatabase

with GraphDatabase() as session:
    result = matgraphdb.execute_query(query, parameters)
```

**Filter properties**
```python
from matgraphdb import GraphDatabase

with MatGraphDB() as session:
    results=session.read_material(
                            material_ids=['mp-1000','mp-1001'], 
                            elements=['Te','Ba'])
    results=session.read_material(
                                material_ids=['mp-1000','mp-1001'],
                                elements=['Te','Ba'], 
                                crystal_systems=['cubic'])

    results=session.read_material(
                            material_ids=['mp-1000','mp-1001'],
                            elements=['Te','Ba'],
                            crystal_systems=['hexagonal'])
                            
    results=session.read_material(
                            material_ids=['mp-1000','mp-1001'],
                            elements=['Te','Ba'],
                            hall_symbols=['Fm-3m'])

    results=session.read_material(
                            material_ids=['mp-1000','mp-1001'],
                            elements=['Te','Ba'],
                            band_gap=[(1.0,'>')])

success,failed=db.check_property(property_name="band_gap")

```


## Authors
[List of contributors and maintainers of the project.]


## Contributing
[Guidelines for contributing to the project, including coding standards, pull request processes, etc.]

