# Graph_Network_Project
This project is for training models encode polyhedron into a vector




# Putting data into Neo4j

The best way to put large amounts of data into neo4j is to use the **neo4j-admin** to import csv files when the dbms is not runnning. This allows to put data into the database in a parallel way without having to worry abour ACID operations
To put nodes and relationships csv's produced into neo 4j follow these steps:

1. Make sure the dbms you want to put the database in is not running.
2. Open the terminal for the particular dbms.
3. run the following command
```bash
.\bin\neo4j-admin.bat import --database test --nodes import\elements.csv --relationships import\Element_Element.csv
```
4. Start the dbms. Then create a new databse with same name as in the previous command. "test"
