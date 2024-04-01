# 
This project is for training models encode polyhedron into a vector




# Putting data into Neo4j

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


## Creating vector index on an embedding of a material 

```bash
CREATE VECTOR INDEX `material-MEGNET-embeddings`
FOR (n :Material) ON (n.`MEGNet-MP-2018`) 
OPTIONS {indexConfig:{
    `vector.dimensions`:160,
    `vector.similarity_function`:'cosine'
}}
```

Check if vector indexing worked

```bash
SHOW VECTOR INDEXES YIELD name, type, labelsOrTypes, properties, options
```

Example of how to call similar nodes for a given node

```bash
MATCH (m: Material {`material_id`:'mp-1000'})
CALL db.index.vector.queryNodes('material-MEGNET-embeddings', 10, m.`MEGNet-MP-2018`)
YIELD node as similarMaterial, score
RETURN m,similarMaterial, score
```