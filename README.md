# MatGraphDB

## Introduction to MatGraphDB

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

Navigate to the root directory `MatGraphDB`. Then do the following
#### Use if you are going to not use graph-tool library
**Windows**
```bash
conda env create -f env_win.yml
```

**Linux**
```bash
conda env create -f env_linux.yml
```

#### Use if you are going to not use graph-tool library
This allows you to use the graph-tool library. Currently, we only support the graph-tool library for linux.

**Linux**
```bash
conda env create -f env_graph_tool.yml
```

#### To activeate the enviornment use:

```bash
conda activate matgraphdb
```

### Adjusting configs

Th configurations of the project are stored in the `MatGraphDB/config.yml` file. You can adjust the configurations to your needs. The most important configurations that need to be adjusted are `DB_NAME`, `USER`, `PASSWORD`, `LOCATION`, `NEO4J_DESKTOP_DIR`, and `N_CORES`.



- `DB_NAME`: The name of the database that will be created. This will search for the database with the same name in the `MatGraphDB/data/production` directory.

- `USER`: The username for the Neo4j database.

- `PASSWORD`: The password for the Neo4j database.

- `LOCATION`: This is the location of the Neo4j DBMS. This is usually `"bolt://localhost:7687"`

- `NEO4J_DESKTOP_DIR`: This is the directory where the Neo4j Desktop is installed. This could be in various locations depending on your system.

    - **Windows** - `C:/Users/{username}/.Neo4jDesktop`
    - **Linux** - `/home/neo4j` : Might be different depending on your system

- `N_CORES`: The number of cores to be used for parallel processing.

### Neo4jDektop
To use neo4j, you will need to install the neo4j desktop application. You can download the application from the [neo4j website](https://neo4j.com/docs/operations-manual/current/installation/). Create a project and then create a new database management system (DBMS) , name it `MatGraphDB` and select the `Neo4j Community Edition` as the DBMS.

You will also need to install the APOC library and Graph Data Science Library. You can do this by click on you DBMS name and the on the right clicking `Plugins`, then click on the libraries and install them.

You will also need to set an apoc environment variable. You can do this by running the following code:

```python
with Neo4jGraphDatabase() as manager:
    settings={'apoc.export.file.enabled':'true'}
    manager.set_apoc_environment_variables(settings=settings)
```

After running this code, you will need to stop the dbms and restart it.




## Usage

### Interacting with the json database:
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

### Creating Graph Databases
To create graph databases, you can use the `GraphGenerator` class. This class takes in a `from_scratch` parameter, which determines whether to start from scratch or use an existing graph database. The default value is `False`. This class also takes in a `skip_main_init` parameter, which determines whether to skip the initial node and relationship creation. The default value is `True`.

When the object is created, it will create the main graph database based on the json files in the `MatGraphDB/data/production/json_database` directory. The main graph database will contain the initial material nodes and relationships. The file can be found at `MatGraphDB/data/production/graph_database/main/neo4j_csv` 

```python
from matgraphdb.graph.graph_generator import GraphGenerator

generator=GraphGenerator(skip_main_init=False)
```

Once the initial graph database is created, you can screen the existing materials using the `screen_graph_database` function.

```python
generator.screen_graph_database('nelements-2-2',nelements=(2,2), from_scratch=True)
generator.screen_graph_database('nelements-3-3',nelements=(3,3), from_scratch=True)

generator.screen_graph_database('spg-145',space_groups=[145], from_scratch=True)
generator.screen_graph_database('spg-145-196',space_groups=[145,196], from_scratch=True)
generator.screen_graph_database('spg-no-145',space_groups=[145], from_scratch=True, include=False)
generator.screen_graph_database('spg-no-196',space_groups=[196], from_scratch=True, include=False)

generator.screen_graph_database('elements-no-Ti',elements=["Ti"], from_scratch=True, include=False)
generator.screen_graph_database('elements-no-Fe',elements=["Fe"], from_scratch=True, include=False)
generator.screen_graph_database('elements-no-Ti-Fe',elements=["Ti","Fe"], from_scratch=True, include=False)
```

Here, we are using the `screen_graph_database` function to create a 9 new graph databases. The `nelements` parameter specifies the number of elements to include in the graph database. The `space_groups` parameter specifies the space groups to include in the graph database. The `elements` parameter specifies the elements to include in the graph database. The `from_scratch` parameter determines whether to start from scratch or use an existing graph database. The `include` parameter determines whether to include the specified elements or space groups in the graph database.


### Writing GraphML
To write a graphml file from the graph, you can use the `write_graphml` function. This function takes a graph database name as input and writes the graph to a file in the specified format.

```python
generator.write_graphml(graph_dirname='nelements-2-2')
```


### Interacting with the Graph Databse in Neo4j

**List Database Schema**
```python
from matgraphdb import Neo4jGraphDatabase

with Neo4jGraphDatabase() as session:
    schema_list=session.list_schema()
```

**Execute Cypher Statement**
```python
with Neo4jGraphDatabase() as session:
    result = matgraphdb.query(query, parameters)
```

**Filter properties**
```python
with Neo4jGraphDatabase() as session:
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



### Interacting with the Neo4j Graph Datascience Library

**Initializing the Neo4jGDSManager**

```python
from matgraphdb import Neo4jGraphDatabase,Neo4jGDSManager

with Neo4jGraphDatabase() as session:
    manager=Neo4jGDSManager(session)
```

**Listing the graphs that are loaded into the gds system for a given database**

```python
with Neo4jGraphDatabase() as session:
    manager=Neo4jGDSManager(session)
    database_name=
    results=manager.list_graphs(database_name='main')
    print(results)

```

**Check if graph is in memory**
```python
with Neo4jGraphDatabase() as session:
    manager=Neo4jGDSManager(session)
    results=manager.is_graph_in_memory(database_name='main', graph_name='materials_chemenvElements')
    print(results)
```
**Loading a graph into the gds system**
```python
with Neo4jGraphDatabase() as session:
    manager=Neo4jGDSManager(session)

    database_name='main'
    graph_name='materials_chemenvElements'
    node_projections=['ChemenvElement','Material']
    relationship_projections={
                "GEOMETRIC_ELECTRIC_CONNECTS": {
                "orientation": 'UNDIRECTED',
                "properties": 'weight'
                },
                "COMPOSED_OF": {
                    "orientation": 'UNDIRECTED',
                    "properties": 'weight'
                }
            }
    manager.load_graph_into_memory(database_name=database_name,
                                       graph_name=graph_name,
                                       node_projections=node_projections,
                                       relationship_projections=relationship_projections)
    print(manager.get_graph_info(database_name=database_name,graph_name=graph_name))
```

**Dropping a graph from memory**
```python
with Neo4jGraphDatabase() as session:
    manager=Neo4jGDSManager(session)
    database_name='main'
    graph_name='materials_chemenvElements'
    reuslts=manager.drop_graph(database_name,graph_name)
```

**Using graph algorithms**
Make sure the graph is loaded into memory before running the algorithms.
```python
with Neo4jGraphDatabase() as session:
    manager=Neo4jGDSManager(session)
    database_name='main'
    graph_name='materials_chemenvElements'
    results=manager.run_fastRP_algorithm(database_name=database_name,
                                  graph_name=graph_name,
                                  algorithm_name='pageRank',
                                  algorithm_mode='stream',
                                  embedding_dimension=128,
                                  concurrency=4,
                                  random_seed=42)
    print(results)
```

**Write to graph database**

```python
with Neo4jGraphDatabase() as session:
    manager=Neo4jGDSManager(session)
    database_name='main'
    graph_name='materials_chemenvElements'
    results=manager.run_fastRP_algorithm(database_name=database_name,
                                  graph_name=graph_name,
                                  algorithm_name='pageRank',
                                  algorithm_mode='write',
                                  embedding_dimension=128,
                                  concurrency=4,
                                  random_seed=42,
                                  write_property='fastrp-embedding')
    print(results)
```
**or**

```python
with Neo4jGraphDatabase() as session:
    manager=Neo4jGDSManager(session)
    database_name='main'
    graph_name='materials_chemenvElements'
    results=manager.run_fastRP_algorithm(database_name=database_name,
                                  graph_name=graph_name,
                                  algorithm_name='pageRank',
                                  algorithm_mode='mutate',
                                  embedding_dimension=128,
                                  concurrency=4,
                                  random_seed=42,
                                  mutate_property='fastrp-embedding')
    print(results)

    manager.write_graph(database_name=database_name,
                        graph_name=graph_name,
                        node_properties=['fastrp-embedding'],
                        node_labels=['Materials'],
                        concurrency=4)

    
```

**Export graph to csv**
```python
with Neo4jGraphDatabase() as session:
    manager=Neo4jGDSManager(session)
    database_name='main'
    graph_name='materials_chemenvElements'
    results=manager.export_graph_csv(database_name=database_name,
                                  graph_name=graph_name,
                                  export_name='materials-chemenvElements.csv',
                                  concurrency=4,
                                  default_relationship_type='COMPOSED_OF',
                                  additional_node_properties=['ChemenvElement','Material'])
    print(results)
```


## Authors
Logan Lang,
Aldo Romero,
Eduardo Hernandez,




