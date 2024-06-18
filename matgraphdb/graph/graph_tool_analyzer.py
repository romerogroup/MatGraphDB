import os
import shutil
from glob import glob
from typing import List, Tuple, Union

import pandas as pd
import graph_tool as gt

from matgraphdb import DBManager
from matgraphdb.utils import (GRAPH_DIR,LOGGER)


class GraphToolAnalyzer:
    def __init__(self,graphml_file,db_manager=DBManager()):
        """
        Initializes the GraphToolAnalyzer object.

        Args:
            db_manager (DBManager,optional): The database manager object. Defaults to DBManager().

        """
        self.db_manager=db_manager
        self.graph = gt.load_graph(graphml_file)

    def compute_graph_entropy(self,**kwargs):
        """
        Computes the graph entropy for a given graphml file.

        Args:
            graphml_file (str): The path to the graphml file.
            from_scratch (bool,optional): If True, deletes the graph database and recreates it from scratch. Defaults to False.

        Returns:
            float: The graph entropy.
        """
        
        return None
    
    def plot_graph(self,filename,**kwargs):
        """
        Plots a graph using the graphml file.

        Args:
            graphml_file (str): The path to the graphml file.
            filename (str): The name of the file to save the plot.

        Returns:
            None
        """
        gt.draw.graph_draw(self.graph,output=filename,**kwargs)
        return None
    
if __name__=='__main__':
    analyzer=GraphToolAnalyzer()
    # analyzer.compute_graph_entropy(graphml_file="data/production/materials_project/graph_database/nelements-2-2/nelements-2-2.graphml")


    analyzer.plot_graph(graphml_file="data/production/materials_project/graph_database/nelements-2-2/nelements-2-2.graphml",
                        filename="data/production/materials_project/graph_database/nelements-2-2/nelements-2-2.png")