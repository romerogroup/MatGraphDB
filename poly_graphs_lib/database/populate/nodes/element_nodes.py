from neo4j import GraphDatabase
import pymatgen.core as pmat

from poly_graphs_lib.database import PASSWORD,DBMS_NAME,LOCATION,DB_NAME,CIF_DIR
from poly_graphs_lib.database.populate.nodes import Node

def populate_element_nodes(elements):
    elements = dir(pmat.periodic_table.Element)[:-4]
    create_statements = []
    for element in elements[:]:
        # pymatgen object. Given element string, will have useful properties 
        pmat_element = pmat.periodic_table.Element(element)

        node=Node(node_name=element,class_name='Element')
        node.add_property(name='atomic_number',value=pmat_element.Z)
        node.add_property(name='atomic_mass',value=float(pmat_element.atomic_mass))

        if str(pmat_element.X) != 'nan':
            node.add_property(name='X',value=pmat_element.X)
        else:
            node.add_property(name='X',value=None)

        if str(pmat_element.atomic_radius) != 'nan' and str(pmat_element.atomic_radius) != 'None':
            node.add_property(name='atomic_radius',value=float(pmat_element.atomic_radius))
        else:
            node.add_property(name='atomic_radius',value=None)
        
        if str(pmat_element.group) != 'nan':
            node.add_property(name='group',value=pmat_element.group)
        else:
            node.add_property(name='group',value=None)

        if str(pmat_element.row) != 'nan':
            node.add_property(name='row',value=pmat_element.row)
        else:
            node.add_property(name='row',value=None)

        execute_statement=node.final_execute_statement()
        create_statements.append(execute_statement)

    return create_statements


def main():
    # This statement Connects to the database server
    connection = GraphDatabase.driver(LOCATION, auth=(DBMS_NAME, PASSWORD))
    # To read and write to the data base you must open a session
    session = connection.session(database=DB_NAME)

    elements = dir(pmat.periodic_table.Element)[:-4]
    create_statements=populate_element_nodes(elements=elements)
    for execute_statment in create_statements:
        session.run(execute_statment)

    session.close()
    connection.close()

if __name__ == '__main__':
    main()