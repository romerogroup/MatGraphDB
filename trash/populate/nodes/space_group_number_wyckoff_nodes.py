from matgraphdb.database.neo4j.populate.nodes import Node
from matgraphdb.database.neo4j.node_types import SPG_WYCKOFFS
from matgraphdb.database.neo4j.utils import execute_statements

def populate_spg_wyckoff_nodes(node_names=SPG_WYCKOFFS, class_name='spg_wyckoff'):
    create_statements = []
    for node_name in node_names:
        
        node=Node(node_name=node_name,class_name=class_name)
        execute_statement=node.final_execute_statement()
        create_statements.append(execute_statement)
    return create_statements

def main():
    create_statements=populate_spg_wyckoff_nodes(node_names=SPG_WYCKOFFS)
    execute_statements(create_statements)