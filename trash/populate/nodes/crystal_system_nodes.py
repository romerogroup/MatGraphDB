from matgraphdb.database.neo4j.populate.nodes import Node
from matgraphdb.database.neo4j.node_types import CRYSTAL_SYSTEMS
from matgraphdb.database.neo4j.utils import execute_statements

def populate_crystal_system_nodes(crystal_systems=CRYSTAL_SYSTEMS):
    class_name='crystal_system'
    create_statements = []
    for node_name in crystal_systems:
        
        node=Node(node_name=node_name,class_name=class_name)
        execute_statement=node.final_execute_statement()
        create_statements.append(execute_statement)
    return create_statements

def main():
    create_statements=populate_crystal_system_nodes(crystal_systems=CRYSTAL_SYSTEMS)
    execute_statements(create_statements)

if __name__ == '__main__':
    main()