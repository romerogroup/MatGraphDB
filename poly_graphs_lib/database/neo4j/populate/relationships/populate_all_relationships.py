
from .chemenvElement_multi_connections import populate_chemenvElement_relationships
from .structure_connections import populate_structure_relationships
from .element_connections import populate_element_relationships
from .chemenv_connections import populate_chemenv_relationships

from poly_graphs_lib.database.neo4j.utils import execute_statements


DEFAULT_POPULATE_FUNCTIONS=[populate_chemenvElement_relationships,
                            populate_structure_relationships,
                            populate_element_relationships
                            ]

def populate_all_relationships(default_populate_functions=DEFAULT_POPULATE_FUNCTIONS):
    for default_populate_function in default_populate_functions:
        create_statments=default_populate_function()
        execute_statements(create_statments)
        
def main():
    populate_all_relationships(DEFAULT_POPULATE_FUNCTIONS)

if __name__ =='__main__':
    main()