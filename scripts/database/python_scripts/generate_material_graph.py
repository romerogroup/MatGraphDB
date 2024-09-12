import os

from sandbox.matgraphdb.graph_kit.graphs import MaterialGraph
from matgraphdb.utils import GRAPH_DIR

def main():

    material_graph=MaterialGraph(graph_dir=os.path.join(GRAPH_DIR,'main'))

    print(material_graph.list_relationships())
    print(material_graph.list_nodes())
    
    
if __name__ == '__main__':
    main()