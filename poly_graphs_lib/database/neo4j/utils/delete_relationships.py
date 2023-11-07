from poly_graphs_lib.database.neo4j.utils.execute_statements import execute_statements

def delete_relationships():
    execute_statment = ["MATCH ()-[r]-() DELETE r"]
    execute_statements(execute_statment,n_cores=1)

if __name__ == '__main__':
    delete_relationships()