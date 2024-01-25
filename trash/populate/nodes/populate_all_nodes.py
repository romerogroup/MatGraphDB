from matgraphdb.database.neo4j.populate.nodes.chemenv_nodes import populate_chemenv_nodes
from matgraphdb.database.neo4j.populate.nodes.chemenvElement_nodes import populate_chemenvElement_nodes
from matgraphdb.database.neo4j.populate.nodes.crystal_system_nodes import populate_crystal_system_nodes
from matgraphdb.database.neo4j.populate.nodes.element_nodes import populate_element_nodes
from matgraphdb.database.neo4j.populate.nodes.magnetic_state_nodes import populate_magnetic_state_nodes
from matgraphdb.database.neo4j.populate.nodes.material_nodes import populate_material_nodes
from matgraphdb.database.neo4j.populate.nodes.space_group_number_nodes import populate_spg_nodes
from matgraphdb.database.neo4j.utils import execute_statements


DEFAULT_POPULATE_FUNCTIONS=[populate_chemenv_nodes,
               populate_chemenvElement_nodes,
               populate_crystal_system_nodes,
               populate_element_nodes,
               populate_magnetic_state_nodes,
               populate_material_nodes,
               populate_spg_nodes]

def populate_all_nodes(default_populate_functions=DEFAULT_POPULATE_FUNCTIONS):
    for default_populate_function in default_populate_functions:
        create_statments=default_populate_function()
        execute_statements(create_statments)
        
def main():
    populate_all_nodes(DEFAULT_POPULATE_FUNCTIONS)

if __name__ =='__main__':
    main()