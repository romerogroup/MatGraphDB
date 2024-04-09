from neo4j import GraphDatabase

from matgraphdb.utils import PASSWORD,USER,LOCATION,GRAPH_DB_NAME
from matgraphdb.graph.similarity_chat import get_similarity_query

class MatGraphDB:
    def __init__(self, uri=LOCATION, user=USER, password=PASSWORD):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def __enter__(self):
        self.create_driver()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def create_driver(self):
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        if self.driver:
            self.driver.close()

    def list_schema(self):
        """
        Retrieves the schema of the graph database.

        Returns:
            schema_list (list): A list of strings representing the schema of the graph database.
        """
        schema_list=[]

        # Query for node labels and properties
        labels_query =  "CALL db.schema.nodeTypeProperties()"
        labels_results = self.execute_query(labels_query)
        node_and_properties = {}
        for record in labels_results:

            node_type = record["nodeType"]
            propert_name = record["propertyName"]

            if isinstance(record["propertyTypes"], list):
                property_type = record["propertyTypes"][0]
            else:
                property_type = record["propertyTypes"]
            property_type=property_type.replace("`",'').replace("String",'str').replace('Integer','int').replace('Float','float')
            if node_type not in node_and_properties:
                node_and_properties[node_type] = {propert_name : property_type}
            else:
                node_and_properties[node_type].update({propert_name : property_type})

        # Query for relationship types
        labels_query =  "CALL db.schema.visualization()"
        labels_results = self.execute_query(labels_query)

        for i,record in enumerate(labels_results):
            
            # Getting node types and names
            try:
                node_1_type=record["nodes"][0]['name']
                node_2_type=record["nodes"][1]['name']
                node_1_name=f':`{node_1_type}`'
                node_2_name=f':`{node_2_type}`'
            except:
                raise Exception("Only one node in this graph gb")

            # Adding indexes and contraints
            # node_and_properties[node_1_name].update({'indexes' : type(record["nodes"][0]._properties['indexes']).__name__})
            # node_and_properties[node_2_name].update({'indexes' : type(record["nodes"][1]._properties['indexes']).__name__})

            # node_and_properties[node_1_name].update({'constraints' : type(record["nodes"][0]._properties['constraints']).__name__})
            # node_and_properties[node_2_name].update({'constraints' : type(record["nodes"][1]._properties['constraints']).__name__})

            node_and_properties[node_1_name].update({'indexes' : record["nodes"][0]._properties['indexes']})
            node_and_properties[node_2_name].update({'indexes' : record["nodes"][1]._properties['indexes']})

            node_and_properties[node_1_name].update({'constraints' : record["nodes"][0]._properties['constraints']})
            node_and_properties[node_2_name].update({'constraints' : record["nodes"][1]._properties['constraints']})

            # Get relationship infor for all relationships
            for relationship in record["relationships"]:

                # Get start and end node names
                start_node=relationship.start_node
                end_node=relationship.end_node

                start_node_name=f':`{start_node._properties["name"]}`'
                end_node_name=f':`{end_node._properties["name"]}`'

                # Get relationship type
                relationship_type = relationship.type

                # Get the relationship properties
                query_relationship=f'MATCH ({start_node_name})-[r:`{relationship_type}`]-({end_node_name}) RETURN r LIMIT 1'
                try:
                    relationship = self.execute_query(query_relationship)[0][0]
                
                    relationship_properties = {}
                    for key, value in relationship._properties.items():
                        relationship_properties[key] = type(value).__name__

                    # Create the final schema
                    query_relationship=f'({start_node_name} {node_and_properties[node_1_name]} )-[r:`{relationship_type}` {relationship_properties}]-({end_node_name} {node_and_properties[node_2_name]}) '
                    schema_list.append(query_relationship)
                except:
                    continue
        
        return schema_list
    
    def execute_query(self, query, parameters=None):

        with self.driver.session(database=GRAPH_DB_NAME) as session:
            results = session.run(query, parameters)
            return [record for record in results]

    def execute_llm_query(self, prompt, n_results=5):

        embedding, execute_statement = get_similarity_query(prompt)
        parameters =  {"embedding": embedding,"nresults":n_results}

        with self.driver.session(database=GRAPH_DB_NAME) as session:
            results = session.run(execute_statement, parameters)

            return [record for record in results]


if __name__ == "__main__":

    with MatGraphDB() as session:
        # result = matgraphdb.execute_query(query, parameters)
        schema_list=session.list_schema()

        print(schema_list)
#         prompt = "What are materials similar to the composition TiAu"
#         # prompt = "What are materials with TiAu"
#         # prompt = "What are materials with TiAu"

#         # prompt = "What are some cubic materials"
# #     # prompt = "What are some materials with a large band gap?"
#         prompt = "What are materials with a band_gap greater than 1.0?"
#         results=session.execute_llm_query(prompt,n_results=10)

#         for result in results:
#             print(result['sm']["name"])
#             print(result['sm']["formula_pretty"])
#             print(result['sm']["symmetry"])
#             print(result['score'])
#             # print(results['sm']["band_gap"])
#             print("_"*200)






    # connection = GraphDatabase.driver(LOCATION, auth=(USER, PASSWORD))
    # session = connection.session(database=DB_NAME)

    # results = session.run("MATCH (n:Material) RETURN n LIMIT 25")
    # for record in results:
    #     print(record)
    # session.close()
    # connection.close()