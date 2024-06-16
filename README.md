# MatGraphDB



## Introduction
# Introduction to MatGraphDB

Welcome to **MatGraphDB**, a powerful Python package designed to interface with primary and graph databases for advanced material analysis. MatGraphDB excels in managing vast datasets of materials data, performing complex computational tasks, and encoding material properties and relationships within a graph-based analytical model.

MatGraphDB is structured around several modular components that work together to streamline data management and analysis:

- **DataManager**: Handles interactions with JSON databases and manages the extraction of information from completed calculation directories.
- **CalcManager**: Manages Density Functional Theory (DFT) calculations, including setting up directories and launching calculations within `MaterialsData`.
- **GraphDBGenerator**: Facilitates the creation of nodes and relationships for graph databases, storing this information in a specified directory and generating necessary CSV files for nodes and relationships.
- **Neo4jManager** and **Neo4jGDSManager**: Manage connections to Neo4j databases, allowing for the creation, update, and removal of databases on the Neo4j server, as well as interaction with the Neo4j Graph Data Science library for advanced graph analytics.

The ultimate goal of MatGraphDB is to leverage the capabilities of graph databases, specifically Neo4j, to enable advanced analysis and discovery in the realm of material science. By integrating data management, DFT calculations, and graph database functionalities, MatGraphDB provides a cohesive workflow for researchers and analysts to explore and understand complex material data.

This documentation provides an overview of the package, detailing how the various components interact to facilitate efficient data management, computation, and analysis, ensuring that you can make the most out of your material science research with MatGraphDB.

## Components and Their Interactions

![System Architecture of MatGraphDB](MatGraphDB.png)
*Figure 1: MatGraphDB Package Overview - This diagram illustrates the main components and their interactions within the MatGraphDB package. It highlights the initialization of the `DataManager`, the execution of DFT calculations by `CalcManager`, the generation of the graph database using `GraphDBGenerator`, and the management of Neo4j databases through `Neo4jManager` and `Neo4jGDSManager`. The workflow demonstrates how data flows from JSON files to advanced graph analytics, facilitating comprehensive materials data analysis.

### 1. DataManager Initialization
The `DataManager` is the foundational component of the package. It is initialized with the `directory_path`, which points to the JSON database directory. The primary role of `DataManager` is to manage interactions with the JSON files and extract information from the completed calculation directories.

### 2. DFT Calculations with CalcManager
The `CalcManager` component manages DFT calculations through interactions with `data_manager`. This includes:
- Performing DFT calculations (`dft_calcs`).
- Managing calculations within `MaterialsData`.
- Setting up directories and launching calculations.

### 3. Graph Database Generation
The `GraphDBGenerator` plays a crucial role in creating nodes and relationships for the graph database. It utilizes `data_manager` to handle these processes. To build the graph database, `GraphDBGenerator` uses the methods `create_nodes()` and `create_relationships()`. Once the database directory is set up, an additional method transforms the graph database into a GraphML file, compatible with various graph packages.

- Creates and stores information in the `graph_databases/{database_name}/neo4j_csv` directory.
- Generates node CSV files in the format `{node_filename}.csv`.
- Generates relationship CSV files in the format `{node_1_filename}-{node_2_filename}-{connection_names}.csv`.


### 4. Managing Neo4j Database Connections
Neo4j database connections are managed by `Neo4jManager` and `Neo4jGDSManager`:
- **Neo4jManager**: After the `GraphDBGenerator` creates the graph database directory, `Neo4jManager` can be initialized with this directory to manage databases on the Neo4j server. This includes creating, removing, updating, and listing databases, as well as importing all data from the directory.
- **Neo4jGDSManager**: Once the database is imported, `Neo4jGDSManager` can be initialized. This component interacts with the Neo4j Graph Data Science library to perform various operations, such as:
  - Loading graphs into memory.
  - Removing graphs from memory.
  - Writing and exporting graphs.
  - Running algorithms on the graphs.

### Summary

MatGraphDB seamlessly integrates materials data management with advanced graph database functionalities, leveraging DFT calculations and Neo4j database management to provide a robust tool for materials science research. The interactions between `DataManager`, `CalcManager`, `GraphDBGenerator`, `Neo4jManager`, and `Neo4jGDSManager` create a cohesive workflow, from data management and DFT calculations to graph database creation and sophisticated data analysis.




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

### Neo4jDektop Interface
To use neo4jdektop, you will need to install the neo4j desktop application. You can download the application from the [neo4j website](https://neo4j.com/docs/operations-manual/current/installation/). Create a project and then create a new database management system (DBMS) , name it `MatGraphDB` and select the `Neo4j Community Edition` as the DBMS.


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

