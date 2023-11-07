from poly_graphs_lib.database.neo4j.populate.nodes.populate_all_nodes import populate_all_nodes

def main():
    populate_all_nodes(execute_statments=True)
    
if __name__ == '__main__':
    main()