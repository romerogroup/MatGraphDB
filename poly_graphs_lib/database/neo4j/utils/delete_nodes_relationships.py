from poly_graphs_lib.database.neo4j.utils.execute_statements import execute_statements

def delete_nodes_relationships():
    execute_statment = ["MATCH (n) DETACH DELETE n"]
    execute_statements(execute_statment,n_cores=1)

if __name__ == '__main__':
    delete_nodes_relationships()