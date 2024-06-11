from typing import List, Tuple, Union

import numpy as np
from neo4j import GraphDatabase
from pymatgen.core import Structure, Composition

from matgraphdb.utils import PASSWORD,USER,LOCATION,GRAPH_DB_NAME
from matgraphdb.graph.similarity_chat import get_similarity_query

class MatGraphDB:

    def __init__(self, uri=LOCATION, user=USER, password=PASSWORD):
        """
        Initializes a GraphDatabase object.

        Args:
            uri (str): The URI of the graph database.
            user (str): The username for authentication.
            password (str): The password for authentication.
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def __enter__(self):
            """
            Enter method for using the graph database as a context manager.

            This method is called when entering a `with` statement and is responsible for setting up any necessary resources
            or connections. In this case, it creates a driver for the graph database and returns the current instance.

            Returns:
                self: The current instance of the `GraphDatabase` class.

            Example:
                with GraphDatabase() as graph_db:
                    # Perform operations on the graph database
            """
            self.create_driver()
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
            """
            Context manager exit method.
            
            This method is called when exiting the context manager. It is responsible for closing the database connection.
            
            Args:
                exc_type (type): The type of the exception raised, if any.
                exc_val (Exception): The exception raised, if any.
                exc_tb (traceback): The traceback object associated with the exception, if any.
            """
            self.close()

    def create_driver(self):
            """
            Creates a driver object for connecting to the graph database.

            Returns:
                None
            """
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
            """
            Closes the database connection.

            This method closes the connection to the database if it is open.
            """
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
            """
            Executes a query on the graph database.

            Args:
                query (str): The Cypher query to execute.
                parameters (dict, optional): Parameters to pass to the query. Defaults to None.

            Returns:
                list: A list of records returned by the query.
            """
            with self.driver.session(database=GRAPH_DB_NAME) as session:
                results = session.run(query, parameters)
                return [record for record in results]
            
    def query(self, query, parameters=None):
            """
            Executes a query on the graph database.

            Args:
                query (str): The Cypher query to execute.
                parameters (dict, optional): Parameters to pass to the query. Defaults to None.

            Returns:
                list: A list of records returned by the query.
            """
            with self.driver.session(database=GRAPH_DB_NAME) as session:
                results = session.run(query, parameters)
            return list(results)
    def execute_llm_query(self, prompt, n_results=5):
        """
        Executes a query in the graph database using the LLM (Language-Modeling) approach.

        Args:
            prompt (str): The prompt for the query.
            n_results (int, optional): The number of results to return. Defaults to 5.

        Returns:
            list: A list of records returned by the query.
        """
        embedding, execute_statement = get_similarity_query(prompt)
        parameters = {"embedding": embedding, "nresults": n_results}

        with self.driver.session(database=GRAPH_DB_NAME) as session:
            results = session.run(execute_statement, parameters)

            return [record for record in results]

    def read_material(self, 
                        material_ids:List[str]=None, 
                        elements:List[str]=None,
                        crystal_systems:List[str]=None,
                        magentic_states:List[str]=None,
                        hall_symbols:List[str]=None,
                        point_groups:List[str]=None,
                        band_gap:List[Tuple[float,str]]=None,
                        vbm:List[Tuple[float,str]]=None,
                        k_vrh:List[Tuple[float,str]]=None,
                        k_voigt:List[Tuple[float,str]]=None,
                        k_reuss:List[Tuple[float,str]]=None,

                        g_vrh:List[Tuple[float,str]]=None,
                        g_voigt:List[Tuple[float,str]]=None,
                        g_reuss:List[Tuple[float,str]]=None,
                        universal_anisotropy:List[Tuple[float,str]]=None,

                        density_atomic:List[Tuple[float,str]]=None,
                        density:List[Tuple[float,str]]=None,
                        e_ionic:List[Tuple[float,str]]=None,
                        
                        e_total:List[Tuple[float,str]]=None,
                        
                        energy_per_atom:List[Tuple[float,str]]=None,

                        compositons:List[str]=None):
            """
            Retrieves materials from the database based on specified criteria.

            Args:
                material_ids (List[str], optional): List of material IDs to filter the results. Defaults to None.
                elements (List[str], optional): List of elements to filter the results. Defaults to None.
                crystal_systems (List[str], optional): List of crystal systems to filter the results. Defaults to None.
                magentic_states (List[str], optional): List of magnetic states to filter the results. Defaults to None.
                hall_symbols (List[str], optional): List of Hall symbols to filter the results. Defaults to None.
                point_groups (List[str], optional): List of point groups to filter the results. Defaults to None.
                band_gap (List[Tuple[float,str]], optional): List of tuples representing the band gap values and comparison operators to filter the results. Defaults to None.
                vbm (List[Tuple[float,str]], optional): List of tuples representing the valence band maximum values and comparison operators to filter the results. Defaults to None.
                k_vrh (List[Tuple[float,str]], optional): List of tuples representing the K_vrh values and comparison operators to filter the results. Defaults to None.
                k_voigt (List[Tuple[float,str]], optional): List of tuples representing the K_voigt values and comparison operators to filter the results. Defaults to None.
                k_reuss (List[Tuple[float,str]], optional): List of tuples representing the K_reuss values and comparison operators to filter the results. Defaults to None.
                g_vrh (List[Tuple[float,str]], optional): List of tuples representing the G_vrh values and comparison operators to filter the results. Defaults to None.
                g_voigt (List[Tuple[float,str]], optional): List of tuples representing the G_voigt values and comparison operators to filter the results. Defaults to None.
                g_reuss (List[Tuple[float,str]], optional): List of tuples representing the G_reuss values and comparison operators to filter the results. Defaults to None.
                universal_anisotropy (List[Tuple[float,str]], optional): List of tuples representing the universal anisotropy values and comparison operators to filter the results. Defaults to None.
                density_atomic (List[Tuple[float,str]], optional): List of tuples representing the atomic density values and comparison operators to filter the results. Defaults to None.
                density (List[Tuple[float,str]], optional): List of tuples representing the density values and comparison operators to filter the results. Defaults to None.
                e_ionic (List[Tuple[float,str]], optional): List of tuples representing the ionic energy values and comparison operators to filter the results. Defaults to None.
                e_total (List[Tuple[float,str]], optional): List of tuples representing the total energy values and comparison operators to filter the results. Defaults to None.
                energy_per_atom (List[Tuple[float,str]], optional): List of tuples representing the energy per atom values and comparison operators to filter the results. Defaults to None.
                compositons (List[str], optional): List of compositions to filter the results. Defaults to None.

            Returns:
                results: The materials that match the specified criteria.
            """
            
            query = f"MATCH (m:Material) WHERE "
            conditional_query = ""

            if material_ids:
                conditional_query += f"m.material_id IN {material_ids}"
            if elements:
                for i,element in enumerate(elements):
                    if len(conditional_query)!=0:
                        conditional_query += " AND "
                    conditional_query += f"'{element}' IN m.elements"
            if crystal_systems:
                if len(conditional_query)!=0:
                    conditional_query += " AND "
                conditional_query += f"m.crystal_system IN {crystal_systems}"
            if magentic_states:
                if len(conditional_query)!=0:
                    conditional_query += " AND "
                conditional_query += f"m.ordering IN {magentic_states}"
            if hall_symbols:
                if len(conditional_query)!=0:
                    conditional_query += " AND "
                conditional_query += f"m.hall_symbol IN {hall_symbols}"
            if point_groups:
                if len(conditional_query)!=0:
                    conditional_query += " AND "
                conditional_query += f"m.point_group IN {point_groups}"

            if band_gap:
                for bg in band_gap:
                    if len(conditional_query)!=0:
                        conditional_query += " AND "
                    value=bg[0]
                    comparison_operator=bg[1]
                    condition_string=f"m.band_gap {comparison_operator} {value}"
                    conditional_query += condition_string

            query += conditional_query
            query +=" RETURN m"
            results = self.execute_query(query)
            return results
    



if __name__ == "__main__":

    with GraphDatabase() as session:
        # result = matgraphdb.execute_query(query, parameters)
        schema_list=session.list_schema()

        # results=session.read_material(material_ids=['mp-1000','mp-1001'],
        #                        elements=['Te','Ba'])
        # results=session.read_material(material_ids=['mp-1000','mp-1001'],
        #                        elements=['Te','Ba'],
        #                        crystal_systems=['cubic'])
        # results=session.read_material(material_ids=['mp-1000','mp-1001'],
        #                        elements=['Te','Ba'],
        #                        crystal_systems=['hexagonal'])
        # results=session.read_material(material_ids=['mp-1000','mp-1001'],
        #                        elements=['Te','Ba'],
        #                        hall_symbols=['Fm-3m'])
        # results=session.read_material(material_ids=['mp-1000','mp-1001'],
        #                        elements=['Te','Ba'],
        #                        band_gap=[(1.0,'>')])
        
        # results=session.create_material(composition="BaTe")

        # print(results)
    #     print(schema_list)
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