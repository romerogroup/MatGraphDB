from pymatgen.core.periodic_table import Element

from poly_graphs_lib.database.neo4j.utils import execute_statements
from poly_graphs_lib.database.neo4j.populate.nodes import Node
from poly_graphs_lib.database.neo4j.populate.nodes.node_types import CHEMENV_ELEMENT_NAMES

def populate_chemenvElement_nodes(class_names=CHEMENV_ELEMENT_NAMES):
    class_name='chemenvElement'
    create_statements = []
    for node_name in class_names:
        node_name=node_name.replace(':','_')
        element_name=node_name.split('_')[0]
        pmat_element = Element(element_name)
        node=Node(node_name=node_name,class_name=class_name)


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
    create_statements=populate_chemenvElement_nodes(class_names=CHEMENV_ELEMENT_NAMES)
    execute_statements(create_statements)

if __name__ == '__main__':
    main()