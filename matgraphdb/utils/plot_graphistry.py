from py2neo import Graph
import graphistry
import pandas as pd

from matgraphdb.database import PASSWORD,DBMS_NAME,LOCATION,DB_NAME

# Initialize Graphistry and Neo4j
# graphistry.register(key='YOUR_GRAPHISTRY_API_KEY')


from neo4j import GraphDatabase, Driver
graphistry.register(bolt=GraphDatabase.driver(LOCATION, auth=(DBMS_NAME, PASSWORD)))
# 
g=graphistry.cypher("MATCH (n1)-[r1]-(n2) RETURN n1, r1, n2 LIMIT 1000")
g.plot()
# # Fetch data from Neo4j (Replace Cypher query as needed)
# cypher_query = """
# MATCH (n)-[r]->(m)
# RETURN n.name AS source, m.name AS destination, r.type AS relationship
# """

# data = neo4j_graph.run(cypher_query).to_data_frame()

# # Convert the data to an edge dataframe
# edge_df = pd.DataFrame(data)

# # Add a dummy property for demonstration (Replace this with actual node property)
# edge_df['property'] = [1, 5, 2, 6, 4, 3]

# # Create an interactive Graphistry plot
# plotter = graphistry.bind(source='source', destination='destination', edge_weight='property')

# # Use the `plotter` object for more complex visualizations and filtering
# # For example, you could use an interactive slider to filter edges based on 'property'

# # Create and display the plot
# plotter.plot(edge_df)