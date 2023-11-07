import numpy as np

from poly_graphs_lib.database.neo4j.populate.nodes import Node
from poly_graphs_lib.database.neo4j.populate.nodes.node_types import SPG_NAMES
from poly_graphs_lib.database.neo4j.utils import execute_statements

def populate_spg_nodes(node_names=SPG_NAMES):
    class_name='space_group'
    create_statements = []
    for node_name in node_names:
        
        node=Node(node_name=node_name,class_name=class_name)
        execute_statement=node.final_execute_statement()
        create_statements.append(execute_statement)
    return create_statements

def main():
    create_statements=populate_spg_nodes(node_names=SPG_NAMES)
    execute_statements(create_statements)


if __name__ == '__main__':
    main()