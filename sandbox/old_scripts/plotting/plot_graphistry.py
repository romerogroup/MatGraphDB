from py2neo import Graph
import graphistry
import pandas as pd

from matgraphdb.database import PASSWORD,DBMS_NAME,LOCATION,DB_NAME


from py2neo import Graph
import graphistry
import pandas as pd
from neo4j import GraphDatabase

# Initialize Graphistry and Neo4j
def initialize_graphistry(api_key: str = None):
    """
    Initialize Graphistry with API key if provided.
    Otherwise, register with Neo4j credentials.
    """
    if api_key:
        graphistry.register(key=api_key)
    else:
        graphistry.register(bolt=GraphDatabase.driver(LOCATION, auth=(DBMS_NAME, PASSWORD)))

# Run Cypher query and fetch graph from Neo4j
def fetch_graph_from_neo4j(cypher_query: str):
    """
    Fetch graph data from Neo4j based on a provided Cypher query.
    
    Args:
        cypher_query (str): The Cypher query to run in Neo4j.
        
    Returns:
        pd.DataFrame: A dataframe containing the query results.
    """
    neo4j_graph = Graph(LOCATION, auth=(DBMS_NAME, PASSWORD))
    data = neo4j_graph.run(cypher_query).to_data_frame()
    return pd.DataFrame(data)

# Plot the fetched graph using Graphistry
def plot_graphistry_graph(cypher_query: str, property_column: str = 'property'):
    """
    Fetch data from Neo4j, add a dummy property, and plot using Graphistry.
    
    Args:
        cypher_query (str): The Cypher query to fetch the data.
        property_column (str): The name of the column to use for edge weight (optional).
    """
    # Fetch data from Neo4j
    edge_df = fetch_graph_from_neo4j(cypher_query)
    
    # Add dummy property or customize it based on the query
    edge_df[property_column] = [1, 5, 2, 6, 4, 3]  # Modify as needed
    
    # Create and display the plot using Graphistry
    plotter = graphistry.bind(source='source', destination='destination', edge_weight=property_column)
    plotter.plot(edge_df)

# Wrapper function to execute everything
def create_and_plot_graph(api_key=None, cypher_query=None, property_column='property'):
    """
    Wrapper function to initialize Graphistry, fetch data, and plot the graph.
    
    Args:
        api_key (str): Optional Graphistry API key for initialization.
        cypher_query (str): Optional Cypher query to fetch data from Neo4j.
        property_column (str): Column used for edge weight in Graphistry (optional).
    """
    initialize_graphistry(api_key)
    
    if cypher_query is None:
        # Default Cypher query if not provided
        cypher_query = """
        MATCH (n)-[r]->(m)
        RETURN n.name AS source, m.name AS destination, r.type AS relationship
        """
    
    plot_graphistry_graph(cypher_query, property_column)
