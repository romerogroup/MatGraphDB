from poly_graphs_lib.database.neo4j.populate.nodes import Node
from poly_graphs_lib.database.neo4j.populate.nodes.node_types import CHEMENV_NAMES
from poly_graphs_lib.database.neo4j.utils import execute_statements

def populate_chemenv_nodes(chemenv_names=CHEMENV_NAMES):
    class_name='chemenv'
    create_statements = []
    for node_name in chemenv_names:
        node_name=node_name.replace(':','_')
        
        node=Node(node_name=node_name,class_name=class_name)

        execute_statement=node.final_execute_statement()

        create_statements.append(execute_statement)
    return create_statements

def main():
    create_statements=populate_chemenv_nodes(chemenv_names=CHEMENV_NAMES)
    execute_statements(create_statements)


if __name__ == '__main__':
    main()