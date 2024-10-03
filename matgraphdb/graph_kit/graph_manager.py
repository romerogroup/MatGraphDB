
from glob import glob
import os
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging

from matgraphdb import config
from matgraphdb.utils.chem_utils.periodic import get_group_period_edge_index
# from matgraphdb.utils.chem_utils.coord_geometry import mp_coord_encoding

from matgraphdb.graph_kit.metadata import get_node_schema,get_relationship_schema
from matgraphdb.graph_kit.metadata import NodeTypes, RelationshipTypes

from matgraphdb.graph_kit.nodes import NodeManager
from matgraphdb.graph_kit.relationships import RelationshipManager
logger = logging.getLogger(__name__)

# TODO: Add screen_graph method
class GraphManager:
    def __init__(self, graph_dir, 
                 output_format='pandas', 
                 node_dirname='nodes', 
                 relationship_dirname='relationships'):
        """
        Initialize the GraphManager with node and relationship directories.

        Args:
            graph_dir (str): The directory where the graph data (nodes and relationships) are stored.
            output_format (str, optional): The format in which to retrieve nodes and relationships (default is 'pandas').
            node_dirname (str, optional): The directory name where node files are stored (default is 'nodes').
            relationship_dirname (str, optional): The directory name where relationship files are stored (default is 'relationships').

        Attributes:
            node_manager (NodeManager): An instance of the NodeManager responsible for managing nodes.
            relationship_manager (RelationshipManager): An instance of the RelationshipManager responsible for managing relationships.
            output_format (str): The format for outputting data (e.g., 'pandas').
        """
        node_dir=os.path.join(graph_dir,node_dirname)
        relationship_dir=os.path.join(graph_dir,relationship_dirname)

        self.node_manager = NodeManager(node_dir, output_format)
        self.relationship_manager = RelationshipManager(relationship_dir, output_format)
        self.output_format = output_format
        
    def get_all_nodes(self):
        """
        Retrieve all existing nodes using the NodeManager.

        Returns:
            list: A list of all existing nodes.
        """
        return self.node_manager.get_existing_nodes()

    def get_all_relationships(self):
        """
        Retrieve all existing relationships using the RelationshipManager.

        Returns:
            list: A list of all existing relationships.
        """
        return self.relationship_manager.get_existing_relationships()

    def get_node(self, node_type):
        """
        Retrieve a specific node by its type.

        Args:
            node_type (str): The type of the node to retrieve.

        Returns:
            pd.DataFrame or dict: The data associated with the requested node type, in the specified output format (e.g., 'pandas').
        """
        return self.node_manager.get_node(node_type, output_format=self.output_format)

    def get_relationship(self, relationship_type):
        """
        Retrieve a specific relationship by its type.

        Args:
            relationship_type (str): The type of the relationship to retrieve.

        Returns:
            pd.DataFrame or dict: The data associated with the requested relationship type, in the specified output format (e.g., 'pandas').
        """
        return self.relationship_manager.get_relationship(relationship_type, output_format=self.output_format)

    def add_node(self, node_class):
        """
        Add a new node using the NodeManager.

        Args:
            node_class (Node): An instance of the Node class to add to the graph.
        """
        self.node_manager.add_node(node_class)

    def add_relationship(self, relationship_class):
        """
        Add a new relationship using the RelationshipManager.

        Args:
            relationship_class (Relationship): An instance of the Relationship class to add to the graph.
        """
        self.relationship_manager.add_relationship(relationship_class)

    def delete_node(self, node_type):
        """
        Delete a node by its type using the NodeManager.

        Args:
            node_type (str): The type of the node to delete.
        """
        self.node_manager.delete_node(node_type)

    def delete_relationship(self, relationship_type):
        """
        Delete a relationship by its type using the RelationshipManager.

        Args:
            relationship_type (str): The type of the relationship to delete.
        """
        self.relationship_manager.delete_relationship(relationship_type)

    def check_node_relationship_consistency(self):
        """
        Check if all relationships refer to existing nodes.

        This method verifies if all nodes referenced in the relationships exist in the node manager.
        It identifies any relationships that reference nodes that are missing from the graph.

        Returns:
            set: A set of missing node types that are referenced in relationships but do not exist in the node manager.
        """
        node_types = self.node_manager.get_existing_nodes()
        relationships = self.relationship_manager.get_existing_relationships()

        # You could enhance this by implementing relationship-specific logic 
        # (for example, checking if all node types in relationships exist)
        missing_nodes = set()
        for rel in relationships:
            rel_data = self.relationship_manager.get_relationship_dataframe(rel)
            nodes_in_relationship = set(rel_data['source']).union(set(rel_data['target']))
            missing_nodes.update(nodes_in_relationship - node_types)

        if missing_nodes:
            logger.warning(f"These nodes are missing in the relationships: {missing_nodes}")
        else:
            logger.info("All relationships are consistent with the nodes.")

        return missing_nodes

    def export_graph_to_neo4j(self, save_dir):
        """
        Export both nodes and relationships to Neo4j CSV format.

        This method exports the graph data (nodes and relationships) to the specified directory in a format compatible
        with Neo4j, allowing the graph to be imported into a Neo4j database.

        Args:
            save_dir (str): The directory where the Neo4j CSV files will be saved.
        """
        os.makedirs(save_dir, exist_ok=True)
        logger.info("Converting all nodes to Neo4j CSV format.")
        self.node_manager.convert_all_to_neo4j(save_dir)

        logger.info("Converting all relationships to Neo4j CSV format.")
        self.relationship_manager.convert_all_to_neo4j(save_dir)

        logger.info(f"Graph successfully exported to Neo4j CSV in {save_dir}")

    def visualize_graph(self):
        """
        Visualize the graph using NetworkX and Matplotlib.

        This method creates a visual representation of the graph by adding nodes and relationships
        to a NetworkX graph and then displaying it using Matplotlib.

        If the required libraries (NetworkX, Matplotlib) are not installed, the method will log an error.

        Raises:
            ImportError: If NetworkX or Matplotlib is not installed.
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt

            G = nx.Graph()

            # Add nodes to the graph
            for node_type in self.node_manager.get_existing_nodes():
                node_df = self.node_manager.get_node_dataframe(node_type)
                for index, row in node_df.iterrows():
                    G.add_node(row['name'], type=node_type)

            # Add relationships to the graph
            for rel_type in self.relationship_manager.get_existing_relationships():
                rel_df = self.relationship_manager.get_relationship_dataframe(rel_type)
                for index, row in rel_df.iterrows():
                    G.add_edge(row['source'], row['target'], relationship=rel_type)

            # Draw the graph
            plt.figure(figsize=(10, 8))
            nx.draw(G, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_weight="bold")
            plt.show()

        except ImportError:
            logger.error("NetworkX and/or matplotlib is not installed. Please install them to visualize the graph.")
    
    def summary(self):
        """
        Return a summary of the current state of the graph.

        The summary includes the total number of nodes and relationships currently present in the graph.

        Returns:
            dict: A dictionary containing the counts of nodes and relationships in the graph.
        """
        node_count = len(self.get_all_nodes())
        relationship_count = len(self.get_all_relationships())
        logger.info(f"Graph contains {node_count} nodes and {relationship_count} relationships.")
        return {
            "nodes": node_count,
            "relationships": relationship_count
        }
