from poly_graphs_lib.database.neo4j.populate.relationships.populate_all_relationships import populate_all_relationships

def main():
    populate_all_relationships(execute_statments=True)
    
if __name__ == '__main__':
    main()