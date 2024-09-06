
from glob import glob
import os
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging

from matgraphdb.utils.periodic_table import get_group_period_edge_index
from matgraphdb.utils.coord_geom import mp_coord_encoding
from matgraphdb.utils import MATERIAL_PARQUET_FILE
from matgraphdb.utils import GRAPH_DIR,PKG_DIR, get_logger
from matgraphdb.graph_kit.metadata import get_node_schema,get_relationship_schema
from matgraphdb.graph_kit.metadata import NodeTypes, RelationshipTypes

from matgraphdb.graph_kit.nodes import NodeManager
from matgraphdb.graph_kit.relationships import RelationshipManager
logger = logging.getLogger(__name__)

# TODO: Add screen_graph method
class GraphManager:
    def __init__(self, node_dir, relationship_dir, output_format='pandas'):
        """
        Initialize the GraphManager with node and relationship directories.
        """
        self.node_manager = NodeManager(node_dir, output_format)
        self.relationship_manager = RelationshipManager(relationship_dir, output_format)
        self.output_format = output_format
        
    def get_all_nodes(self):
        """
        Retrieve all existing nodes using the NodeManager.
        """
        return self.node_manager.get_existing_nodes()

    def get_all_relationships(self):
        """
        Retrieve all existing relationships using the RelationshipManager.
        """
        return self.relationship_manager.get_existing_relationships()

    def get_node(self, node_type):
        """
        Retrieve a specific node by its type.
        """
        return self.node_manager.get_node(node_type, output_format=self.output_format)

    def get_relationship(self, relationship_type):
        """
        Retrieve a specific relationship by its type.
        """
        return self.relationship_manager.get_relationship(relationship_type, output_format=self.output_format)

    def add_node(self, node_class):
        """
        Add a new node using the NodeManager.
        """
        self.node_manager.add_node(node_class)

    def add_relationship(self, relationship_class):
        """
        Add a new relationship using the RelationshipManager.
        """
        self.relationship_manager.add_relationship(relationship_class)

    def delete_node(self, node_type):
        """
        Delete a node by its type using the NodeManager.
        """
        self.node_manager.delete_node(node_type)

    def delete_relationship(self, relationship_type):
        """
        Delete a relationship by its type using the RelationshipManager.
        """
        self.relationship_manager.delete_relationship(relationship_type)

    def check_node_relationship_consistency(self):
        """
        Check if all relationships refer to existing nodes.
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
        """
        os.makedirs(save_dir, exist_ok=True)
        logger.info("Converting all nodes to Neo4j CSV format.")
        self.node_manager.convert_all_to_neo4j(save_dir)

        logger.info("Converting all relationships to Neo4j CSV format.")
        self.relationship_manager.convert_all_to_neo4j(save_dir)

        logger.info(f"Graph successfully exported to Neo4j CSV in {save_dir}")

    def visualize_graph(self):
        """
        Optional method to visualize the graph (using something like NetworkX and matplotlib).
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
        """
        node_count = len(self.get_all_nodes())
        relationship_count = len(self.get_all_relationships())
        logger.info(f"Graph contains {node_count} nodes and {relationship_count} relationships.")
        return {
            "nodes": node_count,
            "relationships": relationship_count
        }
