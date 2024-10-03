import os
import json
from typing import List, Tuple, Union
from glob import glob

from neo4j import GraphDatabase
import pandas as pd

from matgraphdb.data.manager import DBManager
from matgraphdb.utils import (PASSWORD,USER,LOCATION,DBMSS_DIR, GRAPH_DIR, LOGGER,MP_DIR)
from matgraphdb.utils.general_utils import get_os
from matgraphdb.graph_kit.neo4j.utils import get_similarity_query, format_projection,format_dictionary,format_list,format_string


# TODO: Think of way to store new node and relationship properties. 
# TODO: For material nodes, we can use DB Manager to store properties back into json database
# TODO: Created method to export node and relationship properties to csv file it does not have the exact same format as the original file
# TODO: FIX FastRP algorithm


class Neo4jManager:

    def __init__(self, graph_dir=None, db_manager=DBManager(), uri=LOCATION, user=USER, password=PASSWORD, from_scratch=False):
        """
        Initializes a MatGraphDB object.

        Args:
            graph_dir (str): The directory where the graph is located.
            uri (str): The URI of the graph database.
            user (str): The username for authentication.
            password (str): The password for authentication.
            from_scratch (bool): Whether to create a new database or use an existing one.
        """

        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.dbms_dir = None
        self.dbms_json = None
        self.db_manager = db_manager
        self.graph_dir=graph_dir
        self.from_scratch=from_scratch
        self.get_dbms_dir()
        self.neo4j_admin_path=None
        self.neo4j_cypher_shell_path=None
        self.get_neo4j_tools_path()
        if graph_dir:
            self.load_graph_database_into_neo4j(graph_dir)

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
            self.driver=None

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
    
    def get_dbms_dir(self):
        """
        Returns the directory where the database management system (DBMS) is located.

        Returns:
            str: The directory where the DBMS is located.
        """
        dbmss_dirs=glob(os.path.join(DBMSS_DIR,'*'))
        for dbms_dir in dbmss_dirs:
            relate_json=os.path.join(dbms_dir,'relate.dbms.json')
            with open(relate_json,'r') as f:
                dbms_info=json.loads(f.read())
            dbms_name=dbms_info['name'].split()[0]
            if dbms_name=='MatGraphDB':
                self.dbms_dir=dbms_dir
                self.dbms_json=relate_json
                return self.dbms_dir
        if self.dbms_dir is None:
             raise Exception("MatGraphDB DBMS is not found. Please create a new DBMS with the name 'MatGraphDB'")

    def get_neo4j_tools_path(self):
        """
        Returns the path to the Neo4j tools.

        Returns:
            str: The path to the Neo4j tools.
        """
        if get_os()=='Windows':
            self.neo4j_admin_path=os.path.join(self.dbms_dir,'bin','neo4j-admin.bat')
            self.neo4j_cypher_shell_path=os.path.join(self.dbms_dir,'bin','cypher-shell.bat')
        else:
            self.neo4j_admin_path=os.path.join(self.dbms_dir,'bin','neo4j-admin')
            self.neo4j_cypher_shell_path=os.path.join(self.dbms_dir,'bin','cypher-shell')
        return self.neo4j_admin_path,self.neo4j_cypher_shell_path
    
    def get_load_statments(self,database_path):
        """
        Returns the load statement.

        Returns:
            str: The load statement for nodes.
        """
        node_statement=" --nodes"
        node_files=glob(os.path.join(database_path,'nodes','*.csv'))
        for node_file in node_files:
            node_statement+=f" \"{node_file}\""
        relationship_files=glob(os.path.join(database_path,'relationships','*.csv'))
        relationship_statement=" --relationships"
        for relationship_file in relationship_files:
            relationship_statement+=f" \"{relationship_file}\""
        statement=node_statement+relationship_statement
        return statement
    
    def list_databases(self):
        """
        Returns a list of databases in the graph database.

        Returns:
            bool: True if the graph database exists, False otherwise.
        """
        if self.driver is None:
            raise Exception("Graph database is not connected. Please connect to the database first.")
        results=self.execute_query("SHOW DATABASES",database_name='system')
        names=[]
        for result in results:
            graph_name=result['name']
            if graph_name not in ['neo4j','system']:
                names.append(graph_name.lower())
        return names

    def create_database(self,database_name):
        """
        Creates a new database in the graph database.

        Args:
            database_name (str): The name of the database to create.
        """
        if self.driver is None:
            raise Exception("Graph database is not connected. Please connect to the database first.")
        self.execute_query(f"CREATE DATABASE `{database_name}`",database_name='system')

    def remove_database(self,database_name):
        """
        Removes a database from the graph database.

        Args:
            database_name (str): The name of the database to remove.
        """
        if self.driver is None:
            raise Exception("Graph database is not connected. Please connect to the database first.")
        self.execute_query(f"DROP DATABASE `{database_name}`",database_name='system')

    def remove_node(self,database_name,node_type,node_properties=None):
        """
        Removes a node from the graph database.

        Args:
            node_type (str): The type of the node to remove.
            node_properties (dict, optional): The properties of the node to remove. Defaults to None.
        """

        if self.driver is None:
            raise Exception("Graph database is not connected. Please connect to the database first.")
        
        if node_properties is None:
            node_properties=""
        else:
            node_properties=format_dictionary(node_properties)

        cypher_statement=f"MATCH (n:{node_type} {node_properties}) "
        cypher_statement+=f"DETACH DELETE n"
        results=self.query(cypher_statement,database_name=database_name)
        return results

    def remove_relationship(self,database_name,relationship_type,relationship_properties=None):
        """
        Removes a relationship from the graph database.

        Args:
            relationship_type (str): The type of the relationship to remove.
            relationship_properties (dict, optional): The properties of the relationship to remove. Defaults to None.
        """

        if self.driver is None:
            raise Exception("Graph database is not connected. Please connect to the database first.")
        
        if relationship_properties is None:
            relationship_properties=""
        else:
            relationship_properties=format_dictionary(relationship_properties)

        cypher_statement=f"MATCH ()-[r:{relationship_type} {relationship_properties}]-() "
        cypher_statement+=f"DETACH DELETE r"
        results=self.query(cypher_statement,database_name=database_name)
        return results
    
    def load_graph_database_into_neo4j(self,database_path,new_database_name=None):
        """
        Loads a graph database into Neo4j.

        Args:
            graph_datbase_path (str): The path to the graph database to load.
        """
        if new_database_name:
            database_name=new_database_name
        else:
            database_name=os.path.basename(database_path)
        database_path=os.path.join(database_path)
        db_names=self.list_databases()
        database_name=database_name.lower()
        if self.from_scratch and database_name in db_names:
            print(f"Removing database {database_name}")
            self.remove_database(database_name)
        db_names=self.list_databases()

        if database_name in db_names:
            raise Exception(f"Graph database {database_name} already exists. " 
                            "It must be removed before loading. " 
                            "Set from_scratch=True to force a new database to be created.")


        import_statment=f'{self.neo4j_admin_path} database import full'
        load_statment=self.get_load_statments(database_path)
        import_statment+=load_statment
        import_statment+=f" --overwrite-destination {database_name}"
        # Execute the import statement
        os.system(import_statment)

        self.create_database(database_name)
        return None

    def does_property_exist(self,database_name,property_name,node_type=None,relationship_type=None):
        """
        Checks if a property exists in a graph database.

        Args:
            database_name (str): The name of the database.
            node_type (str): The type of the node.
            property_name (str): The name of the property.

        Returns:
            bool: True if the property exists, False otherwise.
        """
        if node_type and relationship_type:
            raise Exception("Both node_type and relationship_type cannot be provided at the same time")
        if node_type is None and relationship_type is None:
            raise Exception("Either node_type or relationship_type must be provided")
        if node_type:
            cypher_statement=f"MATCH (n:`{node_type}`)\n"
            cypher_statement+=f"WHERE n.`{property_name}` IS NOT NULL\n"
            cypher_statement+=f"RETURN n LIMIT 1"
        if relationship_type:
            cypher_statement=f"MATCH ()-[r:`{relationship_type}`]-()\n"
            cypher_statement+=f"WHERE r.`{property_name}` IS NOT NULL\n"
            cypher_statement+=f"RETURN r LIMIT 1"
        results=self.query(cypher_statement,database_name=database_name)
        if len(results)==0:
            return False
        return True

    def remove_property(self,database_name,property_name,node_type=None,relationship_type=None):
        """
        Removes a property from a graph database.

        Args:
            database_name (str): The name of the database.
            node_type (str): The type of the node.
            property_name (str): The name of the property.

        Returns:
            None
        """
        if node_type and relationship_type:
            raise Exception("Both node_type and relationship_type cannot be provided at the same time")
        if node_type is None and relationship_type is None:
            raise Exception("Either node_type or relationship_type must be provided")

        if node_type:
            cypher_statement=f"MATCH (n:{node_type})"
            cypher_statement+=f"DETACH DELETE n.`{property_name}`"
            cypher_statement+=f"RETURN n"
        if relationship_type:
            cypher_statement=f"MATCH ()-[r:{relationship_type}]-()"
            cypher_statement+=f"DETACH DELETE r.`{property_name}`"
            cypher_statement+=f"RETURN r"
        self.query(cypher_statement,database_name=database_name)
        return None

    def execute_query(self, query, database_name, parameters=None):
        """
        Executes a query on the graph database.

        Args:
            query (str): The Cypher query to execute.
            database_name (str): The name of the database to execute the query on. Defaults to None.
            parameters (dict, optional): Parameters to pass to the query. Defaults to None.

        Returns:
            list: A list of records returned by the query.
        """
        if self.driver is None:
            raise Exception("Graph database is not connected. Please connect to the database first.")

        with self.driver.session(database=database_name) as session:
            results = session.run(query, parameters)
            return [record for record in results]
            
    def query(self, query, database_name, parameters=None):
        """
        Executes a query on the graph database.

        Args:
            query (str): The Cypher query to execute.
            database_name (str): The name of the database to execute the query on. Defaults to None.
            parameters (dict, optional): Parameters to pass to the query. Defaults to None.

        Returns:
            list: A list of records returned by the query.
        """
        if self.driver is None:
            raise Exception("Graph database is not connected. Please connect to the database first.")
        with self.driver.session(database=database_name) as session:
            results = session.run(query, parameters)
            return [record for record in results]
    
    def execute_llm_query(self, prompt, database_name, n_results=5):
        """
        Executes a query in the graph database using the LLM (Language-Modeling) approach.

        Args:
            prompt (str): The prompt for the query.
            database_name (str): The name of the database to execute the query on.
            n_results (int, optional): The number of results to return. Defaults to 5.

        Returns:
            list: A list of records returned by the query.
        """
        if self.driver is None:
            raise Exception("Graph database is not connected. Please connect to the database first.")
        embedding, execute_statement = get_similarity_query(prompt)
        parameters = {"embedding": embedding, "nresults": n_results}

        with self.driver.session(database=database_name) as session:
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
    
    def create_vector_index(self,
                            database_name:str,
                            property_dimensions:int,
                            similarity_function:str,
                            node_type:str=None,
                            node_property_name:str=None,
                            relationship_type:str=None,
                            relationship_property_name:str=None,
                            index_name:str=None,
                            ):
        """
        Creates a vector index on a graph for either a node or a relationship.
        https://neo4j.com/docs/graph-data-science/current/management-ops/graph-creation/vector-index/

        Args:
            database_name (str): The name of the database.
            property_dimensions (int): The number of dimensions in the vector.
            similarity_function (str): The similarity function to use.
            node_type (str,optional): The type of the nodes. This is optional and if not provided,
            node_property_name (str, optional): The name of the node property. This is optional and if not provided,
            relationship_type (str, optional): The type of the relationships. This is optional and if not provided,
            relationship_property_name (str, optional): The name of the relationship property. This is optional and if not provided,
            index_name (str, optional): The name of the index. This is optional and if not provided, 
            the default property name is used.

        Returns:
            None
        """

        if node_type is None and node_property_name is None and relationship_type is None and relationship_property_name is None:
            raise Exception("Either node_type and node_property_name or relationship_type and relationship_property_name must be provided")
        if node_type is None and node_property_name is not None:
            raise Exception("node_type must be provided if node_property_name is provided")
        if relationship_type is None and relationship_property_name is not None:
            raise Exception("relationship_type must be provided if relationship_property_name is provided")
        if node_property_name is None and node_type is not None:
            raise Exception("node_property_name must be provided if node_type is provided")
        if relationship_property_name is None and relationship_type is not None:
            raise Exception("relationship_property_name must be provided if relationship_type is provided")

        config={}
        config['vector.dimensions']=property_dimensions
        config['vector.similarity_function']=similarity_function

        cypher_statement=f"CREATE VECTOR INDEX {format_string(index_name)} IF NOT EXISTS"

        if node_type is not None:
            cypher_statement+=f"FOR (n :{format_string(node_type)}) ON (n.{format_string(node_property_name)})"

        if relationship_type is not None:
            cypher_statement+=f"FOR ()-[r :{format_string(relationship_type)}]-() ON (r.{format_string(relationship_property_name)})"
        
        cypher_statement+=" OPTIONS {indexConfig:"
        cypher_statement+=f"{format_dictionary(config)}"
        cypher_statement+="}"

        results=self.query(cypher_statement,database_name=database_name)

        outputs=[]
        for result in results:
            output={ key:value for key, value in result.items()}
            outputs.append(output)
        return outputs
    
    def check_vector_index(self,
                            database_name:str,
                            index_name:str,
                            ):
        """
        Checks if a vector index exists on a graph.
        https://neo4j.com/docs/graph-data-science/current/management-ops/graph-creation/vector-index/

        Args:
            database_name (str): The name of the database.
            index_name (str): The name of the index.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        cypher_statement=f"CALL db.index.list('{format_string(index_name)}')"
        results=self.query(cypher_statement,database_name=database_name)
        if len(results)!=0:
            return True
        return False
    
    def drop_vector_index(self,
                            database_name:str,
                            index_name:str,
                            ):
        """
        Drops a vector index on a graph.
        https://neo4j.com/docs/graph-data-science/current/management-ops/graph-creation/vector-index/

        Args:
            database_name (str): The name of the database.
            index_name (str): The name of the index.

        Returns:
            None
        """
        cypher_statement=f"CALL db.index.drop('{format_string(index_name)}')"
        self.query(cypher_statement,database_name=database_name)
        return None
        
    def query_vector_index(self,
                            database_name:str,
                            graph_name:str,
                            index_name:str,
                            nearest_neighbors:int=10,
                            node_type:str=None,
                            node_property_name:str=None,
                            node_properties:dict=None,
                            relationship_type:str=None,
                            relationship_property_name:str=None,
                            relationship_properties:dict=None,
                            ):
        """
        Queries a vector index on a graph for either a node or a relationship.
        https://neo4j.com/docs/graph-data-science/current/management-ops/graph-creation/vector-index/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            index_name (str): The name of the index.
            nearest_neighbors (int, optional): The number of nearest neighbors to return. Defaults to 10.
            node_type (str,optional): The type of the nodes. This is optional and if not provided,
            node_property_name (str, optional): The name of the node property for which the vector index is queried.
            relationship_type (str, optional): The type of the relationships. This is optional and if not provided,
            relationship_property_name (str, optional): The name of the relationship property. This is optional and if not provided,

        Returns:
            list: A list of tuples representing the query results.
        """
        if node_type is None and node_property_name is None and relationship_type is None and relationship_property_name is None:
            raise Exception("Either node_type and node_property_name or relationship_type and relationship_property_name must be provided")
        if node_type is None and node_property_name is not None:
            raise Exception("node_type must be provided if node_property_name is provided")
        if relationship_type is None and relationship_property_name is not None:
            raise Exception("relationship_type must be provided if relationship_property_name is provided")
        if node_property_name is None and node_type is not None:
            raise Exception("node_property_name must be provided if node_type is provided")
        if relationship_property_name is None and relationship_type is not None:
            raise Exception("relationship_property_name must be provided if relationship_type is provided")
        if node_type is not None and node_properties is None:
            raise Exception("node_properties must be provided. This is to find a specific node")
        if relationship_type is not None and relationship_properties is None:
            raise Exception("relationship_properties must be provided. This is to find a specific relationship")
        
        if node_type:
            cypher_statement=f"MATCH (m :{format_string(node_type)} {format_dictionary(node_properties)})"
            cypher_statement+=f"CALL db.index.vector.queryNodes({format_string(index_name)}, {nearest_neighbors}, m.{format_string(node_property_name)})"
        if relationship_type:
            cypher_statement=f"MATCH ()-[r:{format_string(relationship_type)} {format_dictionary(relationship_properties)}]-()"
            cypher_statement+=f"CALL db.index.vector.queryRelationships({format_string(index_name)}, {nearest_neighbors}, r.{format_string(relationship_property_name)})"
        
        outputs=[]
        for result in self.query(cypher_statement,database_name=database_name):
            output={ key:value for key, value in result.items()}
            outputs.append(output)
        return outputs

    def format_exported_node_file(self,file):
        """
        Formats an exported node file.

        Args:
            file (str): The file to format.

        Returns:
            str: The formatted file.
        """
        df=pd.read_csv(file)
        # df.drop(df.columns[0], axis=1, inplace=True)
        # column_names=list(df.columns)
        # id_col_index=0
        # label_index=0
        # for i,col_name in enumerate(column_names):
        #     if "Id" in col_name:
        #         id_col_index=i
        #     if ':LABEL' in col_name:
        #         label_index=i

        # # Move the id column to the first position and the label to the second position in the dataframe
        # # Get the 'Id' and ':LABEL' columns
        # id_col = df.pop(column_names[id_col_index])
        # label_col = df.pop(column_names[label_index])  # Adjusting index because we've already popped id_col
        # # Insert 'Id' column at the first position
        # df.insert(0, column_names[id_col_index], id_col)
        # # Insert ':LABEL' column at the second position
        # df.insert(1, column_names[label_index], label_col)
        # # Rename id column to 'id'
        # node_name=column_names[id_col_index].split('Id')[0]
        # df.rename(columns={column_names[id_col_index]:f'{column_names[id_col_index]}:ID({node_name}-ID)'}, inplace=True)
        
        return df

    def format_exported_relationship_file(self,file):
        """
        Formats an exported relationship file.

        Args:
            file (str): The file to format.

        Returns:
            str: The formatted file.
        """
        df=pd.read_csv(file)
        # df.drop(df.columns[0], axis=1, inplace=True)
        # column_names=list(df.columns)
        # id_col_index=0
        # label_index=0
        # for i,col_name in enumerate(column_names):
        #     if "Id" in col_name:
        #         id_col_index=i
        #     if ':LABEL' in col_name:
        #         label_index=i

        return df
    
    def export_database(self,
                    graph_dir:str,
                    database_name:str,
                    batch_size:int=20000,
                    delimiter:str=',',
                    array_delimiter:str=';',
                    quotes:str="always",
                    ):
        """
        Exports a database to a file.
        https://neo4j.com/docs/apoc/current/export/csv/

        Args:
            graph_dir (str): The directory of the original graph.
            database_name (str): The name of the database.
            batch_size (int, optional): The batch size. Defaults to 20000.
            delimiter (str, optional): The delimiter. Defaults to ','.
            array_delimiter (str, optional): The array delimiter. Defaults to ';'.
            quotes (str, optional): The quotes. Defaults to 'always'.

        Returns:
            None
        """
        mutated_graphs_dir=os.path.join(graph_dir,'mutated_neo4j_graphs')
        mutated_graph_dir=os.path.join(mutated_graphs_dir,database_name,'neo4j_csv')
        node_dir=os.path.join(mutated_graph_dir,'nodes')
        relationship_dir=os.path.join(mutated_graph_dir,'relationships')
        os.makedirs(node_dir,exist_ok=True)
        os.makedirs(relationship_dir,exist_ok=True)

        config={}
        config['bulkImport']=True
        config['useTypes']=True
        config['batchSize']=batch_size
        config['delimiter']=delimiter
        config['arrayDelimiter']=array_delimiter
        config['quotes']=quotes

        cypher_statement=f"""CALL apoc.export.csv.all("tmp.csv", {format_dictionary(config)})"""
        results=self.query(cypher_statement,database_name=database_name)

        import_dir=os.path.join(self.dbms_dir,"import")
        files=glob(os.path.join(import_dir,'*.csv'))
        for file in files:
            filename=os.path.basename(file)
            _, graph_element, element_type,_=filename.split('.')
            element_type=element_type.lower()
            if 'nodes' == graph_element:
                df=self.format_exported_node_file(file)
                new_file=os.path.join(node_dir,f'{graph_element}.csv')
                df.to_csv(new_file)
            elif 'relationships' == graph_element:
                df=self.format_exported_node_file(file)
                new_file=os.path.join(node_dir,f'{graph_element}.csv')
                df.to_csv(new_file)

        for file in files:
            os.remove(file)
            
        return results

    def set_apoc_environment_variables(self,settings:dict=None, overwrite:bool=False):
        """
        Sets the apoc export file.
        https://neo4j.com/docs/apoc/current/export/csv/

        Args:
            database_name (str): The name of the database.

        Returns:
            None
        """
        
        conf_file=os.path.join(self.dbms_dir,'conf','apoc.conf')
        # Reading the existing config
        with open(conf_file, 'r') as file:
            lines = file.readlines()
        
        # Modifying the config
        with open(conf_file, 'w') as file:
            for key, value in settings.items():
                new_line = f'{key}={value}\n'
                for line in lines:
                    if key in line and overwrite:
                        new_line = f'{key}={value}\n'
                    else:
                        new_line = line
                    
                file.write(new_line)
        return None
    

# if __name__=='__main__':
    # with Neo4jManager() as manager:
    #     results=manager.does_property_exist('elements-no-fe','Material','fastrp-embedding')
    #     print(results)

    # with Neo4jManager() as manager:
    #     results=manager.does_property_exist('elements-no-fe','Material','fastrp-embedding')
    #     print(results)