from poly_graphs_lib.database.neo4j.populate.nodes import Node
from poly_graphs_lib.database.neo4j.populate.nodes.node_types import MAGNETIC_STATES
from poly_graphs_lib.database.neo4j.utils import execute_statements

def populate_magnetic_state_nodes(node_names=MAGNETIC_STATES):
    class_name='magnetic_states'
    create_statements = []
    for node_name in node_names:
        node=Node(node_name=node_name,class_name=class_name)
        execute_statement=node.final_execute_statement()
        create_statements.append(execute_statement)
    return create_statements

def main():
    create_statements=populate_magnetic_state_nodes(node_names=MAGNETIC_STATES)
    execute_statements(create_statements)

if __name__ == '__main__':
    main()