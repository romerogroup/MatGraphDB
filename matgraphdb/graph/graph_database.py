import os
import json
from typing import List, Tuple, Union
from glob import glob
import textwrap
from neo4j import GraphDatabase

import pandas as pd

from matgraphdb.utils import PASSWORD,USER,LOCATION,GRAPH_DB_NAME,DBMSS_DIR,MAIN_GRAPH_DIR,GRAPH_DIR, LOGGER,MP_DIR
from matgraphdb.utils.general import get_os
from matgraphdb.graph.similarity_chat import get_similarity_query

class MatGraphDB:

    def __init__(self,database_path=None,uri=LOCATION, user=USER, password=PASSWORD, from_scratch=False):
        """
        Initializes a MatGraphDB object.

        Args:
            database_path (str): The path to the database.
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
        self.from_scratch=from_scratch
        self.get_dbms_dir()
        self.neo4j_admin_path=None
        self.neo4j_cypher_shell_path=None
        self.get_neo4j_tools_path()
        if database_path:
            self.load_graph_database_into_neo4j(database_path)

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
    
    def get_databases(self):
        """
        Returns a list of databases in the graph database.

        Returns:
            bool: True if the graph database exists, False otherwise.
        """
        self.create_driver()
        results=self.execute_query("SHOW DATABASES",database_name='system')
        names=[result['name'] for result in results]
        self.close()
        return names

    def remove_database(self,database_name):
        """
        Removes a database from the graph database.

        Args:
            database_name (str): The name of the database to remove.
        """
        self.create_driver()
        self.execute_query(f"DROP DATABASE `{database_name}`",database_name='system')
        self.close()

    def create_database(self,database_name):
        """
        Creates a new database in the graph database.

        Args:
            database_name (str): The name of the database to create.
        """
        self.create_driver()
        self.execute_query(f"CREATE DATABASE `{database_name}`",database_name='system')
        self.close()

    def load_graph_database_into_neo4j(self,database_path):
        """
        Loads a graph database into Neo4j.

        Args:
            graph_datbase_path (str): The path to the graph database to load.
        """
        database_name=os.path.basename(database_path)
        db_names=self.get_databases()
        if self.from_scratch and database_name in db_names:
            self.remove_database(database_name)
        db_names=self.get_databases()

        if database_name in db_names:
            raise Exception(f"Graph database {database_name} already exists. " 
                            "It must be removed before loading. " 
                            "Set from_scratch=True to force a new database to be created.")


        import_statment=f'{self.neo4j_admin_path} database import full'
        load_statment=self.get_load_statments(database_path)
        import_statment+=load_statment
        import_statment+=f" --overwrite-destination {database_name}"
        print(import_statment)
        # Execute the import statement
        os.system(import_statment)

        self.create_database(database_name)
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
    
class Neo4jDLManager:
    def __init__(self, matgraphdb:MatGraphDB):
        self.matgraphdb = matgraphdb
        if self.matgraphdb.driver is None:
            raise Exception("Graph database is not connected. Please ccreate a driver")
        
    def list_graphs(self,database_name):
        """
        Lists the graphs in a database.
        https://neo4j.com/docs/graph-data-science/current/management-ops/graph-list/

        Args:
            database_name (str): The name of the database.
        Returns:
            list: A list of the graphs in the database.
        """
        cypher_statement=f"""
        CALL gds.graph.list()
        YIELD graphName
        RETURN graphName;
        """
        results = self.matgraphdb.query(cypher_statement,database_name=database_name)
        graph_names=[result['graphName'] for result in results]
        return graph_names
    
    def is_graph_in_memory(self,database_name,graph_name):
        """
        Checks if the graph exists in memory.
        https://neo4j.com/docs/graph-data-science/current/management-ops/graph-list/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.

        Returns:
            bool: True if the graph exists in memory, False otherwise.
        """

        cypher_statement=f"""
        CALL gds.graph.list("{graph_name}")
        YIELD graphName
        RETURN graphName;
        """
        results = self.matgraphdb.query(cypher_statement,database_name=database_name)
        if len(results)!=0:
            return True
        return False
    
    def drop_graph(self,database_name,graph_name):
        """
        Drops a graph from a database. 
        https://neo4j.com/docs/graph-data-science/current/management-ops/graph-drop/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.

        Returns:
            None
        """
        cypher_statement=f"""
        CALL gds.graph.drop("{graph_name}")
        """
        self.matgraphdb.query(cypher_statement,database_name=database_name)
        return None

    def list_graph_data_science_algorithms(self,database_name, save=False):
        """
        Lists the algorithms in a database.
        https://neo4j.com/docs/graph-data-science/current/management-ops/graph-list-algorithms/

        Args:
            database_name (str): The name of the database.

        Returns:
            list: A list of the algorithms in the database.
        """
        cypher_statement=f"""
        CALL gds.list()
        YIELD name,description
        RETURN  name,description;
        """
        results = self.matgraphdb.query(cypher_statement,database_name=database_name)
        algorithm_names={result['name']:result['description'] for result in results}
        if save:
            print("Saving algorithms to : ",os.path.join(MP_DIR,'neo4j_graph_data_science_algorithms.txt'))
            with open(os.path.join(MP_DIR,'neo4j_graph_data_science_algorithms.json'), "w") as file:
                json.dump(algorithm_names, file, indent=4)
            # Decide on a fixed width for the name column
            with open(os.path.join(MP_DIR,'neo4j_graph_data_science_algorithms.txt'), "w") as file:
                fmt = '{0:75s}{1:200s}\n'
                for result in results:
                    file.write(fmt.format(result['name'],result['description']))
                    file.write('_'*300)
                    file.write('\n')
        return algorithm_names

    def load_graph_into_memory(self,database_name:str, 
                               graph_name:str,
                               node_projections:Union[str,List,dict], 
                               relationship_projections:Union[str,List,dict],
                               config:dict=None):
        """
        Loads a graph into memory.
        https://neo4j.com/docs/graph-data-science/current/management-ops/graph-creation/graph-project/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            nodes_names (list): A list of node names.
            relationships_names (list): A list of relationship names.

        Returns:
            None
        """
        if self.is_graph_in_memory(database_name=database_name,graph_name=graph_name):
            LOGGER.info(f"Graph {graph_name} already in memory")
            return None
        formated_node_str=format_projection(projections=node_projections)
        formated_relationship_str=format_projection(projections=relationship_projections)
        
        cypher_statement=f"""CALL gds.graph.project(
                "{graph_name}",
                {formated_node_str},
                {formated_relationship_str}"""
        
        if config:
            cypher_statement+=", "
            cypher_statement+=format_dictionary(projections=config)
        cypher_statement+=")"

        self.matgraphdb.query(cypher_statement,database_name=database_name)
        return None
    
    def estimate_memeory_for_algorithm(self,database_name:str,graph_name:str,algorithm_name:str,model:str='stream',algorithm_config:dict=None):
        """
        Estimates the memory required for a given algorithm.
        https://neo4j.com/docs/graph-data-science/current/management-ops/graph-estimate-memory/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_name (str): The name of the algorithm.

        Returns:
            float: The estimated memory required for the algorithm.
        """
        cypher_statement=f"CALL gds.{algorithm_name}.{model}.estimate("
        cypher_statement+=f"{format_string(graph_name)},"
        cypher_statement+=f"{format_dictionary(algorithm_config)}"
        cypher_statement+=")"
        cypher_statement+="YIELD nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory"
        result = self.matgraphdb.query(cypher_statement,database_name=database_name)[0]
        names=('nodeCount','relationshipCount','bytesMin','bytesMax','requiredMemory')
        return {name:result[name] for name in names}
    
def format_list(prop_list):
    """
    Formats a list into a string for use in Cypher queries.

    Args:
        prop_list (list): A list containing the properties.

    Returns:
        str: A string representation of the properties.
    """
    return [f"{prop}" for prop in prop_list]

def format_string(prop_string):
    """
    Formats a string into a string for use in Cypher queries.

    Args:
        prop_string (str): A string containing the properties.

    Returns:
        str: A string representation of the properties.
    """
    return f"'{prop_string}'"

def format_dictionary(prop_dict):
    """
    Formats a dictionary into a string for use in Cypher queries.

    Args:
        prop_dict (dict): A dictionary containing the properties.

    Returns:
        str: A string representation of the properties.
    """
    formatted_properties="{"
    n_props=len(prop_dict)
    for i,(prop_name,prop_params) in enumerate(prop_dict.items()):
        if isinstance(prop_params,str):
            formatted_properties+=f"{prop_name}: {format_string(prop_params)}"
        elif isinstance(prop_params,int):
            formatted_properties+=f"{prop_name}: {prop_params}"
        elif isinstance(prop_params,float):
            formatted_properties+=f"{prop_name}: {prop_params}"
        elif isinstance(prop_params,List):
            formatted_properties+=f"{prop_name}: {format_list(prop_params)}"
        elif isinstance(prop_params,dict):
            formatted_properties+=f"{prop_name}: {format_dictionary(prop_params)}"

        if i!=n_props-1:
            formatted_properties+=", "

    formatted_properties+="}"
    return formatted_properties

def format_projection(projections:Union[str,List,dict]):
    formatted_projections=""
    if isinstance(projections,List):
        formatted_projections=format_list(projections)
    elif isinstance(projections,dict):
        formatted_projections=format_dictionary(projections)
    elif isinstance(projections,str):       
        formatted_projections=format_string(projections)
    return formatted_projections


if __name__ == "__main__":
    # database_path=os.path.join(GRAPH_DIR,'main')
    # db=MatGraphDB(database_path=database_path,from_scratch=True)
    # print(db.get_databases())
    with MatGraphDB() as matgraphdb:
        database_name='nelements-1-2'
        manager=Neo4jDLManager(matgraphdb)
        print(manager.list_graphs(database_name))
        print(manager.is_graph_in_memory(database_name,'materials_chemenvElements'))
        # print(manager.list_graph_data_science_algorithms(database_name))
        graph_name='materials_chemenvElements'
        node_projections=['ChemenvElement','Material']
        relationship_projections={
                    "GEOMETRIC_ELECTRIC_CONNECTS": {
                    "orientation": 'UNDIRECTED',
                    "properties": 'weight'
                    },
                    "COMPOSED_OF": {
                        "orientation": 'UNDIRECTED',
                        "properties": 'weight'
                    }
                }
        
        # print(format_dictionary(relationship_projections))
        manager.load_graph_into_memory(database_name=database_name,
                                       graph_name=graph_name,
                                       node_projections=node_projections,
                                       relationship_projections=relationship_projections)
        
        print(manager.list_graphs(database_name))
        # print(manager.is_graph_in_memory(database_name,graph_name))
        # print(manager.drop_graph(database_name,graph_name))
        # print(manager.list_graphs(database_name))
        # print(manager.is_graph_in_memory(database_name,graph_name))


        result=manager.estimate_memeory_for_algorithm(database_name=database_name,
                                               graph_name=graph_name,
                                               algorithm_name='fastRP',
                                               model='stream',
                                               algorithm_config={'embeddingDimension':128})
        print(result)
    # # database_path=os.path.join(GRAPH_DIR,'nelements-1-2')
    # db=MatGraphDB(database_path=database_path,from_scratch=True)
    # print(db.get_databases())
    # with GraphDatabase() as session:
    #     # result = matgraphdb.execute_query(query, parameters)
    #     schema_list=session.list_schema()

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

    ##################################################################################################
    # # Testing Neo4jDLManager
    ##################################################################################################
    # with MatGraphDB() as matgraphdb:
    #     database_name='nelements-1-2'
    #     manager=Neo4jDLManager(matgraphdb)
    #     print(manager.list_graphs(database_name))
    #     print(manager.is_graph_in_memory(database_name,'materials_chemenvElements'))

    #     graph_name='materials_chemenvElements'
    #     node_projections=['ChemenvElement','Material']
        # node_projections={
        #     "ChemenvElement":{
        #         "label":'ChemenvElement',
        #         "properties":{
        #             "chemical_symbol":{'property':'chemical_symbol','defaultValue':'','aggregation':'sum'},
        #             "atomic_number":{'property':'atomic_number','defaultValue':0,'aggregation':'sum'},
        #             "X":{'property':'X','defaultValue':0.0,'aggregation':'sum'},
        #             "atomic_radius":{'property':'atomic_radius','defaultValue':0.0,'aggregation':'sum'},
        #             "group":{'property':'group','defaultValue':0,'aggregation':'sum'},
        #             "row":{'property':'row','defaultValue':0,'aggregation':'sum'},
        #             "atomic_mass":{'property':'atomic_mass','defaultValue':0.0,'aggregation':'sum'}}
        #         }
        #     }
    #     relationship_projections={
    #                 "GEOMETRIC_ELECTRIC_CONNECTS": {
    #                 "orientation": 'UNDIRECTED',
    #                 "properties": 'weight'
    #                 },
    #                 "COMPOSED_OF": {
    #                     "orientation": 'UNDIRECTED',
    #                     "properties": {
    #                                    'distance':{'property':'distance','defaultValue':0.0,'aggregation':'sum'},
    #                                    'weight':{'property':'weight','defaultValue':1.0,'aggregation':'sum'}
    #                                 }
    #                 }
    #             }
        
    #     print(format_dictionary(relationship_projections))
    #     manager.load_graph_into_memory(database_name=database_name,
    #                                    graph_name=graph_name,
    #                                    node_projections=node_projections,
    #                                    relationship_projections=relationship_projections)

    ##################################################################################################
    # # Testing Neo4jDLManager
    ##################################################################################################
    # with MatGraphDB() as matgraphdb:
    #     database_name='nelements-1-2'
    #     manager=Neo4jDLManager(matgraphdb)
    #     print(manager.list_graphs(database_name))
    #     print(manager.is_graph_in_memory(database_name,'materials_chemenvElements'))
        results=manager.list_graph_data_science_algorithms(database_name,save=True)
        # for result in results:
        #     # print the reuslts in two columns
        #     print(result[0],'|||||||||||||||',result[1])