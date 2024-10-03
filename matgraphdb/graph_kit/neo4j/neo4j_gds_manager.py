import os
import json
from typing import List, Tuple, Union
from glob import glob

from neo4j import GraphDatabase
import pandas as pd

from matgraphdb.utils import (PASSWORD, USER, LOCATION, DBMSS_DIR, GRAPH_DIR, LOGGER,MP_DIR)
from matgraphdb.utils.general_utils import get_os
from matgraphdb.graph_kit.neo4j.utils import format_projection, format_dictionary, format_list, format_string


class Neo4jGDSManager:

    def __init__(self, neo4j_manager):
        self.neo4j_manager = neo4j_manager
        self.algorithm_modes=['stream','stats','write','mutate']
        self.link_prediction_algorithms=['adamicAdar','commonNeighbors','preferentialAttachment',
                                         'resourceAllocation','sameCommunity','totalNeighbors']
        if self.neo4j_manager.driver is None:
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
        results = self.neo4j_manager.query(cypher_statement,database_name=database_name)
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
        results = self.neo4j_manager.query(cypher_statement,database_name=database_name)
        if len(results)!=0:
            return True
        return False
    
    def get_graph_info(self,database_name,graph_name):
        """
        Gets the graph information.
        https://neo4j.com/docs/graph-data-science/current/management-ops/graph-info/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.

        Returns:
            dict: The graph information.
        """

        cypher_statement=f"CALL gds.graph.list(\"{graph_name}\")"
        results = self.neo4j_manager.query(cypher_statement,database_name=database_name)
        outputs=[]
        for result in results:
            output={ key:value for key, value in result.items()}
            outputs.append(output)
        return outputs

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
        self.neo4j_manager.query(cypher_statement,database_name=database_name)
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
        results = self.neo4j_manager.query(cypher_statement,database_name=database_name)
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

    def get_graph_data_science_algorithms_docs_url(self):
        """
        Lists the algorithms in a database.
        https://neo4j.com/docs/graph-data-science/current/management-ops/graph-list-algorithms/

        Args:
            database_name (str): The name of the database.

        Returns:
            list: A list of the algorithms in the database.
        """
        website_url=f"https://neo4j.com/docs/graph-data-science/current/algorithms/"
        return website_url
    
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

        self.neo4j_manager.query(cypher_statement,database_name=database_name)
        return None
    
    def read_graph(self,
                   database_name:str,
                   graph_name:str,
                   node_properties:Union[str,List[str]],
                   node_labels:str=None,
                   concurrency:int=4,
                   list_node_labels:bool=False,
                   ):
        """
        Reads a graph from memory.
        https://neo4j.com/docs/graph-data-science/current/management-ops/graph-creation/read-graph/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            node_properties (str): The name of the node properties.
            algorithm_mode (str, optional): The algorithm mode. Defaults to 'stream'.
            node_labels (str, optional): The name of the node labels. Defaults to None.
            concurrency (int, optional): The concurrency. Defaults to 4.
            list_node_labels (bool, optional): The list node labels. Defaults to False.

        Returns:
            None
        """
        if not self.is_graph_in_memory(database_name=database_name,graph_name=graph_name):
            LOGGER.info(f"Graph {graph_name} not in memory")
            return None
        config={}
        config['concurrency']=concurrency
        config['listNodeLabels']=list_node_labels

        cypher_statement=f"CALL gds.graph.nodeProperties.stream("
        cypher_statement+=f"{format_string(graph_name)},"
        if isinstance(node_properties,str):
            cypher_statement+=f"{format_string(node_properties)}"
        elif isinstance(node_properties,list):
            cypher_statement+=f"{format_list(node_properties)}"
        if node_labels:
            if isinstance(node_labels,str):
                cypher_statement+=f", {format_string(node_labels)}"
            elif isinstance(node_labels,list):
                cypher_statement+=f", {format_list(node_labels)}"
        cypher_statement+=f")"
        print(cypher_statement)
        results=self.neo4j_manager.query(cypher_statement,database_name=database_name)
        return results

    def write_graph(self,
                    database_name:str,
                    graph_name:str,
                    node_properties:Union[str,List,dict]=None,
                    node_labels:Union[str,List[str]]=None,
                    relationship_type:str=None,
                    relationship_properties:Union[str,List[str]]=None,
                    concurrency:int=4):
        """
        Write graph to neo4j database.
        https://neo4j.com/docs/graph-data-science/current/management-ops/graph-creation/graph-project/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            node_properties (str): The node properties.
            node_labels (str): The node labels.
            relationship_properties (str): The relationship properties.
            relationship_labels (str): The relationship labels.
            concurrency (int, optional): The concurrency. Defaults to 4.

        Returns:    
            list: Returns results of the algorithm.
        """
        if not self.is_graph_in_memory(database_name=database_name,graph_name=graph_name):
            raise Exception(f"Graph {graph_name} is not in memory")
        node_outputs=[]
        relationship_outputs=[]

        config={}
        config['concurrency']=concurrency
        config['writeConcurrency']=concurrency
        if node_properties:
            cypher_statement=f"CALL gds.graph.nodeProperties.write("
            cypher_statement+=f"{format_string(graph_name)},"
            cypher_statement+=f"{format_projection(node_properties)},"
            if node_labels:
                cypher_statement+=f"{format_list(node_labels)},"
            cypher_statement+=f"{format_dictionary(config)}"
            cypher_statement+=")"

            results=self.neo4j_manager.query(cypher_statement,database_name=database_name)
            for result in results:
                node_output={property_name:property_value for property_name,property_value in result.items()}
                node_outputs.append(node_output)

        if relationship_type:
            cypher_statement=f"CALL gds.graph.relationshipProperties.write("
            cypher_statement+=f"{format_string(graph_name)},"
            cypher_statement+=f"{format_string(relationship_type)},"   
            if relationship_properties:
                cypher_statement+=f"{format_list(relationship_properties)},"
            cypher_statement+=f"{format_dictionary(config)}"
            cypher_statement+=")"

            results=self.neo4j_manager.query(cypher_statement,database_name=database_name)
            for result in results:
                relationship_output={property_name:property_value for property_name,property_value in result.items()}
                relationship_outputs.append(relationship_output)

        return node_outputs,relationship_outputs
    
    def export_graph(self,
                    database_name:str,
                    graph_name:str,
                    db_name:str,
                    concurrency:int=4,
                    batch_size:int=100,
                    default_relationship_type:str='__ALL__',
                    additional_node_properties:Union[str,List,dict]=None,
                    ):
        """
        Export graph to neo4j database.

        Useful urls.
            https://neo4j.com/docs/graph-data-science/current/management-ops/graph-creation/graph-project/
        
        Useful scenarios.
            - Avoid heavy write load on the operational system by exporting the data instead of writing back.
            - Create an analytical view of the operational system that can be used as a basis for running algorithms.
            - Produce snapshots of analytical results and persistent them for archiving and inspection.
            - Share analytical results within the organization.

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            db_name (str): The name of the database. Defaults to None.
            concurrency (int, optional): The concurrency. Defaults to 4.
            batch_size (int, optional): The batch size. Defaults to 100.
            default_relationship_type (str, optional): The default relationship type. Defaults to '__ALL__'.
            additional_node_properties (str, optional): The additional node properties. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """
        if not self.is_graph_in_memory(database_name=database_name,graph_name=graph_name):
            raise Exception(f"Graph {graph_name} is not in memory")
        if db_name not in self.neo4j_manager.list_databases():
            raise Exception(f"Database {db_name} does not exist")
        
        config={}
        config['dbName']=db_name
        config['concurrency']=concurrency
        config['writeConcurrency']=concurrency
        config['batchSize']=batch_size
        config['defaultRelationshipType']=default_relationship_type
        if additional_node_properties:
            config['additionalNodeProperties']=format_projection(additional_node_properties)
        cypher_statement=f"CALL gds.graph.export("
        cypher_statement+=f"{format_string(graph_name)},"
        cypher_statement+=f"{format_dictionary(config)}"
        cypher_statement+=")"

        

        results=self.neo4j_manager.query(cypher_statement,database_name=database_name)
        outputs=[]
        for result in results:
            output={property_name:property_value for property_name,property_value in result.items()}
            outputs.append(output)
        return outputs

    def export_graph_csv(self,
                        database_name:str,
                        graph_name:str,
                        export_name:str,
                        concurrency:int=4,
                        default_relationship_type:str='__ALL__',
                        additional_node_properties:Union[str,List,dict]=None,
                        use_label_mapping:bool=False,
                        sampling_factor:float=0.001,
                        estimate_memory:bool=False,
                        ):
        """
        Export graph to neo4j database.

        Useful urls.
            https://neo4j.com/docs/graph-data-science/current/management-ops/graph-creation/graph-project/
        
        
        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            export_name (str): The export name. Absolute directory path is required. Defaults to None.
            concurrency (int, optional): The concurrency. Defaults to 4.
            default_relationship_type (str, optional): The default relationship type. Defaults to '__ALL__'.
            additional_node_properties (str, optional): The additional node properties. Defaults to None.
            use_label_mapping (bool, optional): The use label mapping. Defaults to False.

        Returns:    
            list: Returns results of the algorithm.
        """
        if not self.is_graph_in_memory(database_name=database_name,graph_name=graph_name):
            raise Exception(f"Graph {graph_name} is not in memory")
        if export_name is None:
            raise Exception("export_name must be provided")
        
        config={}
        if estimate_memory:
            config['exportName']=export_name
            config['samplingFactor']=sampling_factor
            config['writeConcurrency']=concurrency
            config['defaultRelationshipType']=default_relationship_type
            cypher_statement=f"CALL gds.graph.export.csv.estimate("
            cypher_statement+=f"{format_string(graph_name)},"
            cypher_statement+=f"{format_dictionary(config)}"
            cypher_statement+=")"
        else:
            config['exportName']=export_name
            config['writeConcurrency']=concurrency
            config['defaultRelationshipType']=default_relationship_type
            if additional_node_properties:
                config['additionalNodeProperties']=format_projection(additional_node_properties)
            config['useLabelMapping']=use_label_mapping
            
            cypher_statement=f"CALL gds.graph.export.csv("
            cypher_statement+=f"{format_string(graph_name)},"
            cypher_statement+=f"{format_dictionary(config)}"
            cypher_statement+=")"

        
        results=self.neo4j_manager.query(cypher_statement,database_name=database_name)
        outputs=[]
        for result in results:
            output={property_name:property_value for property_name,property_value in result.items()}
            outputs.append(output)
        return outputs

    def estimate_memeory_for_algorithm(self,database_name:str,
                                       graph_name:str,
                                       algorithm_name:str,
                                       algorithm_mode:str='stream',
                                       algorithm_config:dict=None):
        """
        Estimates the memory required for a given algorithm.
        f"https://neo4j.com/docs/graph-data-science/current/algorithms/"
        https://neo4j.com/docs/graph-data-science/current/management-ops/graph-estimate-memory/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_name (str): The name of the algorithm.

        Returns:
            float: The estimated memory required for the algorithm.
        """
        if not self.is_graph_in_memory(database_name=database_name,graph_name=graph_name):
            raise Exception(f"Graph {graph_name} is not in memory")
        cypher_statement=f"CALL gds.{algorithm_name}.{algorithm_mode}.estimate("
        cypher_statement+=f"{format_string(graph_name)},"
        cypher_statement+=f"{format_dictionary(algorithm_config)}"
        cypher_statement+=")"
        results = self.neo4j_manager.query(cypher_statement,database_name=database_name)
        outputs=[]
        for result in results:
            output={property_name:property_value for property_name,property_value in result.items()}
            outputs.append(output)

        return outputs
    
    def run_algorithm(self,database_name:str,
                      graph_name:str,
                      algorithm_name:str,
                      algorithm_mode:str='stream',
                      algorithm_config:dict=None):
        """
        Estimates the memory required for a given algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/algorithms/
            https://neo4j.com/docs/graph-data-science/current/management-ops/graph-estimate-memory/
            neo4j.com/docs/graph-data-science/current/operations-reference/algorithm-references/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_name (str): The name of the algorithm.

        Returns:
            float: The estimated memory required for the algorithm.
        """
        if not self.is_graph_in_memory(database_name=database_name,graph_name=graph_name):
            raise Exception(f"Graph {graph_name} is not in memory")
        cypher_statement=f"CALL gds.{algorithm_name}.{algorithm_mode}("
        cypher_statement+=f"{format_string(graph_name)},"
        cypher_statement+=f"{format_dictionary(algorithm_config)}"
        cypher_statement+=")"
        results = self.neo4j_manager.query(cypher_statement,database_name=database_name)
        outputs=[]
        for result in results:
            output={property_name:property_value for property_name,property_value in result.items()}
            outputs.append(output)
        return outputs
    
    def run_fastRP_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str='stream',
                            embedding_dimension:int=128,
                            random_seed:int=42,
                            concurrency:int=4,
                            property_ratio:float=0.0,
                            feature_properties:List[str]=[],
                            iteration_weights:List[float]=[0.0,1.0,1.0],
                            node_self_influence:float=0.0,
                            normalization_strength:float=0.0,
                            relationship_weight_property:str=None,
                            write_property:str=None,
                            mutate_property:str=None,
                            ):
        """
        Estimates the memory required for a given algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/node-embeddings/fastrp/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            embedding_dimension (int, optional): The dimension of the embedding. Defaults to 128.
            random_seed (int, optional): The random seed. Defaults to 42.
            concurrency (int, optional): The concurrency. Defaults to 4.
            property_ratio (float, optional): The property ratio. Defaults to 0.0.
            feature_properties (list, optional): The feature properties. Defaults to [].
            iteration_weights (list, optional): The iteration weights. Defaults to [0.0,1.0,1.0].
            node_self_influence (float, optional): The node self influence. Defaults to 0.0.
            normalization_strength (float, optional): The normalization strength. Defaults to 0.0.
            relationship_weight_property (str, optional): The relationship weight property. Defaults to 'null'.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.
        Returns:    
            list: Returns results of the algorithm.
        """
        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        if embedding_dimension is None:
            raise Exception("Embedding_dimension must be provided")
        algorithm_config={}
        algorithm_config['embeddingDimension']=embedding_dimension
        algorithm_config['randomSeed']=random_seed
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        algorithm_config['propertyRatio']=property_ratio
        algorithm_config['featureProperties']=feature_properties
        # algorithm_config['iterationWeights']=iteration_weights
        algorithm_config['nodeSelfInfluence']=node_self_influence
        algorithm_config['normalizationStrength']=normalization_strength
        if relationship_weight_property:
            algorithm_config['relationshipWeightProperty']=relationship_weight_property
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property

        results=self.run_algorithm(database_name=database_name,
                      graph_name=graph_name,
                      algorithm_name='fastRP',
                      algorithm_mode=algorithm_mode,
                      algorithm_config=algorithm_config)
        return results
    
    def run_node2vec_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str='stream',
                            embedding_dimension:int=128,
                            embedding_initializer:str='NORMALIZED',
                            walk_length:int=80,
                            walks_per_node:int=10,
                            in_out_factor:float=1.0,
                            return_factor:float=1.0,
                            relationship_weight_property:str='null',
                            window_size:int=10,
                            negative_sample_rate:int=5,
                            positive_sampling_factor:float=0.001,
                            negative_sampling_exponent:float=0.75,
                            iterations:int=1,
                            initial_learning_rate:float=0.01,
                            min_learning_rate:float=0.0001,
                            walk_buffer_size:int=100,
                            random_seed:int=42,
                            concurrency:int=4,
                            write_property:str='n/a',
                            mutate_property:str=None,
                            ):
        """
        Runs the node2vec algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/node-embeddings/node2vec/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            embedding_dimension (int, optional): The dimension of the embedding. Defaults to 128.
            embedding_initializer (str, optional): The initializer of the embedding. Defaults to 'NORMALIZED'.
            walk_length (int, optional): The length of the walks. Defaults to 80.
            walks_per_node (int, optional): The number of walks per node. Defaults to 10.
            in_out_factor (float, optional): The factor for the in-out ratio. Defaults to 1.0.
            return_factor (float, optional): The factor for the return ratio. Defaults to 1.0.
            relationship_weight_property (str, optional): The relationship weight property. Defaults to 'null'.
            window_size (int, optional): The window size. Defaults to 10.
            negative_sample_rate (int, optional): The negative sample rate. Defaults to 5.
            positive_sampling_factor (float, optional): The positive sampling factor. Defaults to 0.001.
            negative_sampling_exponent (float, optional): The negative sampling exponent. Defaults to 0.75.
            iterations (int, optional): The number of iterations. Defaults to 1.
            initial_learning_rate (float, optional): The initial learning rate. Defaults to 0.01.
            min_learning_rate (float, optional): The minimum learning rate. Defaults to 0.0001.
            walk_buffer_size (int, optional): The walk buffer size. Defaults to 100.
            random_seed (int, optional): The random seed. Defaults to 42.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_property (str, optional): The write property. Defaults to 'n/a'.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """
        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        if embedding_dimension is None:
            raise Exception("Embedding_dimension must be provided")
        
        algorithm_config={}
        algorithm_config['embeddingDimension']=embedding_dimension
        algorithm_config['embeddingInitializer']=embedding_initializer
        algorithm_config['randomSeed']=random_seed
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        algorithm_config['walkLength']=walk_length
        algorithm_config['walksPerNode']=walks_per_node
        algorithm_config['inOutFactor']=in_out_factor
        algorithm_config['returnFactor']=return_factor
        
        algorithm_config['windowSize']=window_size
        algorithm_config['negativeSampleRate']=negative_sample_rate
        algorithm_config['positiveSamplingFactor']=positive_sampling_factor
        algorithm_config['negativeSamplingExponent']=negative_sampling_exponent
        algorithm_config['iterations']=iterations
        algorithm_config['initialLearningRate']=initial_learning_rate
        algorithm_config['minLearningRate']=min_learning_rate
        algorithm_config['walkBufferSize']=walk_buffer_size
        if relationship_weight_property:
            algorithm_config['relationshipWeightProperty']=relationship_weight_property
        if write_property:
            algorithm_config['write_property']=write_property
        if mutate_property:
            algorithm_config['mutate_property']=mutate_property

        results=self.run_algorithm(
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='node2vec',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        
        return results

    def run_hashGNN_algorithm(self,
                              database_name:str,
                              graph_name:str,
                              algorithm_mode:str='stream',
                              feature_properties:List[str]=[],
                              iterations:int=None,
                              embedding_density:int=None,
                              heterogeneous:bool=False,
                              neighbor_influence:float=1.0,
                              binarize_features:dict=None,
                              generate_features:dict=None,
                              output_dimension:int=None,
                              random_seed:int=42,
                              mutate_property:str=None):
        """
        Runs the hashGNN algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/hashgnn/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            algorithm_config (dict, optional): The configuration of the algorithm. Defaults to None.
            embedding_density (int, optional): The embedding density. Defaults to None.
            heterogeneous (bool, optional): Whether the graph is heterogeneous. Defaults to False.
            neighbor_influence (float, optional): The neighbor influence. Defaults to 1.0.
            binarize_features (dict, optional): The binarize features. Defaults to None.
            generate_features (dict, optional): The generate features. Defaults to None.
            output_dimension (int, optional): The output dimension. Defaults to None.
            random_seed (int, optional): The random seed. Defaults to 42.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """
        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        if iterations is None:
            raise Exception("Iterations must be provided")
        if embedding_density is None:
            raise Exception("Embedding density must be provided")
        if generate_features and feature_properties is not None:
            raise Exception("Feature properties must be None when generate_features is given")
        
        algorithm_config={}
        algorithm_config['featureProperties']=feature_properties
        algorithm_config['iterations']=iterations
        algorithm_config['embeddingDensity']=embedding_density
        algorithm_config['heterogeneous']=heterogeneous
        algorithm_config['neighborInfluence']=neighbor_influence
        algorithm_config['randomSeed']=random_seed
        if binarize_features:
            algorithm_config['binarizeFeatures']=binarize_features
        if generate_features:
            algorithm_config['generateFeatures']=generate_features
        if output_dimension:
            algorithm_config['outputDimension']=output_dimension
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property
        
        results=self.run_algorithm(
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='hashgnn',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results

    def run_graphSAGE_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str='stream',
                            model_name:str=None,
                            concurrency:int=4,
                            batch_size:int=100,
                            mutate_property:str=None,
                            write_property:str=None
                            ):
        """
        Runs the graphSAGE algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/graph-sage/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            model_name (str, optional): The model name. Defaults to None.
            concurrency (int, optional): The concurrency. Defaults to 4.
            batch_size (int, optional): The batch size. Defaults to 100.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """
        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        if model_name is None:
            raise Exception("Model name must be provided")
        if write_property:
            raise Exception("write_property must be None")
        
        algorithm_config={}
        algorithm_config['modelName']=model_name
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        algorithm_config['batchSize']=batch_size
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='beta.graphSage',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results

    def run_topological_link_prediction_algorithm(self,
                            database_name:str,
                            algorithm_name:str,
                            node_a_name:str,
                            node_a_type:str,
                            node_b_name:str,
                            node_b_type:str,
                            relationship_query:str=None,
                            direction:str='BOTH',
                            comunity_property:str='community',
                            ):
        """
        Runs the topological link prediction algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/topological-link-prediction/

        Args:
            database_name (str): The name of the database.
            node_a_name (str): The name of the node A.
            node_a_type (str): The type of the node A.
            node_b_name (str): The name of the node B.
            node_b_type (str): The type of the node B.
            relationship_query (str): The relationship query.
            direction (str, optional): The direction. Defaults to 'BOTH'.

        Returns:
            list: Returns results of the algorithm.
        """
        if algorithm_name not in self.link_prediction_algorithms:
            raise Exception(f"Algorithm name {algorithm_name} is not supported. Use one of {self.link_prediction_algorithms}")
        
        config={}
        config['direction']=direction
        if relationship_query:
            config['relationshipQuery']=relationship_query
        if algorithm_name=='SameCommunity':
            config['communityProperty']=comunity_property

        cypher_statement=f"MATCH (p1:{node_a_type}" + "{name:" + f"{format_string(node_a_name)}" + "})\n"
        cypher_statement=f"MATCH (p1:{node_b_type}" + "{name:" + f"{format_string(node_b_name)}" + "})\n" 
        cypher_statement=f"RETURN gds.alpha.linkprediction.{algorithm_name}(p1,p2,{format_dictionary(config)})"
        results = self.neo4j_manager.query(cypher_statement,database_name=database_name)
        outputs=[]
        for result in results:
            output={property_name:property_value for property_name,property_value in result.items()}
            outputs.append(output)
        return outputs
    
    def run_dijkstra_source_target_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            source_node:int,
                            target_node:int,
                            relationship_weight_property:str='weight',
                            concurrency:int=1,
                            write_node_ids:bool=False,
                            write_costs:bool=False,
                            write_relationship_type:str=None,
                            mutate_relationship_type:str=None,
                            ):
        """
        Runs the dijkstra source target algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/dijkstra/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            source_node (int, optional): The source node. Defaults to None.
            target_node (int, optional): The target node. Defaults to None.
            relationship_weight_property (str, optional): The relationship weight property. Defaults to 'weight'.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_node_ids (bool, optional): The write node ids. Defaults to False.
            write_costs (bool, optional): The write costs. Defaults to False.
            write_relationship_type (bool, optional): The write relationship type. Defaults to False.
            mutate_relationship_type (str, optional): The mutate relationship type. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """
        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_relationship_type is None:
            raise Exception("write_relationship_type must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_relationship_type is None:
            raise Exception("mutate_relationship_type must be provided when algorithm_mode is mutate")
        if not isinstance(source_node,int):
            raise Exception("Source node must be an integer")
        if not isinstance(target_node,int):
            raise Exception("Target node must be an integer")

        algorithm_config={}
        algorithm_config['sourceNode']=source_node
        algorithm_config['targetNode']=target_node
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        if write_node_ids:
            algorithm_config['writeNodeIds']=write_node_ids
        if write_costs:
            algorithm_config['writeCosts']=write_costs
        if write_relationship_type:
            algorithm_config['writeRelationshipType']=write_relationship_type
        if mutate_relationship_type:
            algorithm_config['mutateRelationshipType']=mutate_relationship_type
        if relationship_weight_property:
            algorithm_config['relationshipWeightProperty']=relationship_weight_property

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='shortestPath.dijkstra',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
    
    def run_a_star_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            source_node:int,
                            target_node:int,
                            lattitude_property:float,
                            longitude_property:float,
                            write_relationship_type:str=None,
                            relationship_weight_property:str='weight',
                            concurrency:int=1,
                            write_node_ids:bool=False,
                            write_costs:bool=False,
                            mutate_relationship_type:str=None,
                            ):
        """
        Runs the a star algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/a-star/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            source_node (int, optional): The source node. Defaults to None.
            target_node (int, optional): The target node. Defaults to None.
            lattitude_property (float, optional): The lattitude property. Defaults to None.
            longitude_property (float, optional): The longitude property. Defaults to None.
            relationship_weight_property (str, optional): The relationship weight property. Defaults to 'weight'.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_node_ids (bool, optional): The write node ids. Defaults to False.
            write_costs (bool, optional): The write costs. Defaults to False.
            write_relationship_type (bool, optional): The write relationship type. Defaults to False.
            mutate_relationship_type (str, optional): The mutate relationship type. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """
        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_relationship_type is None:
            raise Exception("write_relationship_type must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_relationship_type is None:
            raise Exception("mutate_relationship_type must be provided when algorithm_mode is mutate")
        if not isinstance(source_node,int):
            raise Exception("Source node must be an integer")
        if not isinstance(target_node,int):
            raise Exception("Target node must be an integer")
        if not isinstance(lattitude_property,float):
            raise Exception("Lattitude property must be a float")
        if not isinstance(longitude_property,float):
            raise Exception("Longitude property must be a float")

        algorithm_config={}
        algorithm_config['sourceNode']=source_node
        algorithm_config['targetNode']=target_node
        algorithm_config['lattitudeProperty']=lattitude_property
        algorithm_config['longitudeProperty']=longitude_property
        if relationship_weight_property:
            algorithm_config['relationshipWeightProperty']=relationship_weight_property
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        if write_node_ids:
            algorithm_config['writeNodeIds']=write_node_ids
        if write_costs:
            algorithm_config['writeCosts']=write_costs
        if write_relationship_type:
            algorithm_config['writeRelationshipType']=write_relationship_type
        if mutate_relationship_type:
            algorithm_config['mutateRelationshipType']=mutate_relationship_type

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='shortestPath.astar',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
    
    def run_yens_shortest_path_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            source_node:int,
                            target_node:int,
                            k:int=1,
                            relationship_weight_property:str='weight',
                            concurrency:int=1,
                            write_relationship_type:str=None,
                            write_node_ids:bool=False,
                            write_costs:bool=False,
                            mutate_relationship_type:str=None,
                            ):
        """
        Runs the yen's shortest path algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/yen-shortest-path/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            source_node (int, optional): The source node. Defaults to None.
            target_node (int, optional): The target node. Defaults to None.
            k (int, optional): The k. Defaults to 1.
            relationship_weight_property (str, optional): The relationship weight property. Defaults to 'weight'.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_node_ids (bool, optional): The write node ids. Defaults to False.
            write_costs (bool, optional): The write costs. Defaults to False.
            mutate_relationship_type (str, optional): The mutate relationship type. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """
        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_relationship_type is None:
            raise Exception("write_relationship_type must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_relationship_type is None:
            raise Exception("mutate_relationship_type must be provided when algorithm_mode is mutate")
        if not isinstance(source_node,int):
            raise Exception("Source node must be an integer")
        if not isinstance(target_node,int):
            raise Exception("Target node must be an integer")

        algorithm_config={}
        algorithm_config['sourceNode']=source_node
        algorithm_config['targetNode']=target_node
        algorithm_config['k']=k
        if relationship_weight_property:
            algorithm_config['relationshipWeightProperty']=relationship_weight_property
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        if write_node_ids:
            algorithm_config['writeNodeIds']=write_node_ids
        if write_costs:
            algorithm_config['writeCosts']=write_costs
        if mutate_relationship_type:
            algorithm_config['mutateRelationshipType']=mutate_relationship_type
        if write_relationship_type:
            algorithm_config['writeRelationshipType']=write_relationship_type
        

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='shortestPath.yens',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
    
    def run_node_similarity_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str='stream',
                            similarity_cutoff:float=10**-42,
                            degree_cutoff:int=1,
                            upper_degree_cutoff:int=2147483647,
                            topK:int=10,
                            bottomK:int=10,
                            topN:int=0,
                            bottomN:int=0,
                            relationship_weight_property:str=None,
                            similarity_metric:str='JACCARD',
                            use_components:bool=False,
                            write_relationship_type:str=None,
                            write_property:str=None,
                            mutate_relationship_type:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the node similarity algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/node-similarity/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            similarity_cutoff (float, optional): The similarity cutoff. Defaults to 10**-42.
            degree_cutoff (int, optional): The degree cutoff. Defaults to 1.
            upper_degree_cutoff (int, optional): The upper degree cutoff. Defaults to 2147483647.
            topK (int, optional): The top K. Defaults to 10.
            bottomK (int, optional): The bottom K. Defaults to 10.
            topN (int, optional): The top N. Defaults to 0.
            bottomN (int, optional): The bottom N. Defaults to 0.
            relationship_weight_property (str, optional): The relationship weight property. Defaults to None.
            similarity_metric (str, optional): The similarity metric. Defaults to 'JACCARD'.
            use_components (bool, optional): The use components. Defaults to False.
            write_relationship_type (str, optional): The write relationship type. Defaults to None.
            write_property (str, optional): The write property. Defaults to None.
            mutate_relationship_type (str, optional): The mutate relationship type. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """
        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and (write_property is None or write_relationship_type is None):
            raise Exception("write_property and write_relationship_type must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and (mutate_property is None or mutate_relationship_type is None):
            raise Exception("mutate_property and mutate_relationship_type must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        algorithm_config['similarityCutoff']=similarity_cutoff
        algorithm_config['degreeCutoff']=degree_cutoff
        algorithm_config['upperDegreeCutoff']=upper_degree_cutoff
        algorithm_config['topK']=topK
        algorithm_config['bottomK']=bottomK
        algorithm_config['topN']=topN
        algorithm_config['bottomN']=bottomN
        algorithm_config['similarityMetric']=similarity_metric
        algorithm_config['useComponents']=use_components
        if relationship_weight_property:
            algorithm_config['relationshipWeightProperty']=relationship_weight_property
        if write_relationship_type:
            algorithm_config['writeRelationshipType']=write_relationship_type
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_relationship_type:
            algorithm_config['mutateRelationshipType']=mutate_relationship_type
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='nodeSimilairty',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
    
    def run_k_nearest_neighbors_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            node_properties:Union[str,List,dict],
                            topK:int=10,
                            sample_rate:float=0.5,
                            delta_threshold:float=0.001,
                            max_iterations:int=100,
                            random_joins:int=10,
                            initial_sampler:str='uniform',
                            random_seed:int=42,
                            similarity_cutoff:float=0,
                            perturbation_rate:float=0.0,
                            concurrency:int=1,
                            write_relationship_type:str=None,
                            write_property:str=None,
                            write_node_ids:bool=False,
                            write_costs:bool=False,
                            mutate_relationship_type:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the k nearest neighbors algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/k-nearest-neighbors/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            node_properties (Union[str,List,dict], optional): The node properties. Defaults to None.
            topK (int, optional): The top K. Defaults to 10.
            sample_rate (float, optional): The sample rate. Defaults to 0.5.
            delta_threshold (float, optional): The delta threshold. Defaults to 0.001.
            max_iterations (int, optional): The max iterations. Defaults to 100.
            random_joins (int, optional): The random joins. Defaults to 10.
            initial_sampler (str, optional): The initial sampler. Defaults to 'uniform'.
            random_seed (int, optional): The random seed. Defaults to 42.
            similarity_cutoff (float, optional): The similarity cutoff. Defaults to 0.
            perturbation_rate (float, optional): The perturbation rate. Defaults to 0.0.
            concurrency (int, optional): The concurrency. Defaults to 1.
            write_relationship_type (str, optional): The write relationship type. Defaults to None.
            write_property (str, optional): The write property. Defaults to None.
            write_node_ids (bool, optional): The write node ids. Defaults to False.
            write_costs (bool, optional): The write costs. Defaults to False.
            mutate_relationship_type (str, optional): The mutate relationship type. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.
        
        Returns:    
            list: Returns results of the algorithm.
        """
        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and (write_property is None or write_relationship_type is None):
            raise Exception("write_property and write_relationship_type must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and (mutate_property is None or mutate_relationship_type is None):
            raise Exception("mutate_property and mutate_relationship_type must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        algorithm_config['nodeProperties']=node_properties
        algorithm_config['topK']=topK
        algorithm_config['sampleRate']=sample_rate  
        algorithm_config['deltaThreshold']=delta_threshold
        algorithm_config['maxIterations']=max_iterations
        algorithm_config['randomJoins']=random_joins
        algorithm_config['initialSampler']=initial_sampler
        algorithm_config['randomSeed']=random_seed
        algorithm_config['similarityCutoff']=similarity_cutoff
        algorithm_config['perturbationRate']=perturbation_rate
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        if write_node_ids:
            algorithm_config['writeNodeIds']=write_node_ids
        if write_costs:
            algorithm_config['writeCosts']=write_costs
        if write_relationship_type:
            algorithm_config['writeRelationshipType']=write_relationship_type
        if mutate_relationship_type:
            algorithm_config['mutateRelationshipType']=mutate_relationship_type
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='knn',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
    
    def run_conductance_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            relationship_weight_property:str=None,
                            community_property:str=None,
                            concurrency:int=1,
                            ):
        """
        Runs the conductance algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/conductance/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            relationship_weight_property (str, optional): The relationship weight property. Defaults to None.
            community_property (str, optional): The community property. Defaults to None.
            concurrency (int, optional): The concurrency. Defaults to 1.

        Returns:    
            list: Returns results of the algorithm.
        """
        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        
        algorithm_config={}
        algorithm_config['concurrency']=concurrency
        if relationship_weight_property:
            algorithm_config['relationshipWeightProperty']=relationship_weight_property
        if community_property:
            algorithm_config['communityProperty']=community_property
        
        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='conductance',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results

    def run_k_core_decomposition_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            concurrency:int=1,
                            mutate_property:str=None,
                            write_property:str=None):
        """
        Runs the k core decomposition algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/k-core-decomposition/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            concurrency (int, optional): The concurrency. Defaults to 1.
            write_property (str, optional): The write property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """
        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        algorithm_config['concurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        
        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='kcore',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
    
    def run_k1_coloring_algorithm(self,
                                  database_name:str,
                                  graph_name:str,
                                  algorithm_mode:str,
                                  min_community_size:int=0,
                                  max_iterations:int=10,
                                  concurrency:int=1,
                                  write_property:str=None,
                                  mutate_property:str=None,
                                  ):
        """
        Runs the k1 coloring algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/k1-coloring/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            min_community_size (int, optional): The min community size. Defaults to 0.
            max_iterations (int, optional): The max iterations. Defaults to 10.
            concurrency (int, optional): The concurrency. Defaults to 1.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """
        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and (write_property is None or mutate_property is None):
            raise Exception("write_property and mutate_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and (mutate_property is None or write_property is None):
            raise Exception("mutate_property and write_property must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        algorithm_config['minCommunitySize']=min_community_size
        algorithm_config['maxIterations']=max_iterations
        algorithm_config['concurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        
        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='k1coloring',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results

    def run_label_propagation_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            max_iterations:int=10,
                            node_weight_property:str=None,
                            relationship_weight_property:str=None,
                            seed_property:str=None,
                            consecutive_ids:bool=False,
                            min_community_size:int=0,
                            concurrency:int=4,
                            write_property:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the label propagation algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/label-propagation/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            max_iterations (int, optional): The max iterations. Defaults to 10.
            node_weight_property (str, optional): The node weight property. Defaults to None.
            relationship_weight_property (str, optional): The relationship weight property. Defaults to None.
            seed_property (str, optional): The seed property. Defaults to None.
            consecutive_ids (bool, optional): The consecutive ids. Defaults to False.
            min_community_size (int, optional): The min community size. Defaults to 0.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """
        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property and mutate_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property and write_property must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        algorithm_config['maxIterations']=max_iterations
        if node_weight_property:
            algorithm_config['nodeWeightProperty']=node_weight_property
        if relationship_weight_property:
            algorithm_config['relationshipWeightProperty']=relationship_weight_property
        algorithm_config['seedProperty']=seed_property
        algorithm_config['consecutiveIds']=consecutive_ids
        algorithm_config['minCommunitySize']=min_community_size
        algorithm_config['concurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='labelPropagation',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
    
    def run_leiden_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            max_levels:int=10,
                            gamma:float=1.0,
                            theta:float=0.01,
                            tolerance:float=0.0001,
                            include_intermediate_communities:bool=False,
                            seed_property:str=None,
                            concurrency:int=4,
                            write_property:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the leiden algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/leiden/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            max_levels (int, optional): The max levels. Defaults to 10.
            gamma (float, optional): The gamma. Defaults to 1.0.
            theta (float, optional): The theta. Defaults to 0.01.
            tolerance (float, optional): The tolerance. Defaults to 0.0001.
            include_intermediate_communities (bool, optional): The include intermediate communities. Defaults to False.
            seed_property (str, optional): The seed property. Defaults to None.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """

        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        algorithm_config['maxLevels']=max_levels
        algorithm_config['gamma']=gamma
        algorithm_config['theta']=theta
        algorithm_config['tolerance']=tolerance
        algorithm_config['includeIntermediateCommunities']=include_intermediate_communities
        algorithm_config['seedProperty']=seed_property
        algorithm_config['concurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='leiden',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
    
    def run_local_clustering_coefficient_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            triangle_count_property:str=None,
                            concurrency:int=4,
                            write_property:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the local clustering coefficient algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/local-clustering-coefficient/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            triangle_count_property (str, optional): The triangle count property. Defaults to None.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """

        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        algorithm_config['triangleCountProperty']=triangle_count_property
        algorithm_config['concurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='localClusteringCoefficient',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
    
    def run_louvain_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            relationship_weight_property:str=None,
                            seed_property:str=None,
                            max_levels:int=10,
                            tolerance:float=0.0001,
                            include_intermediate_communities:bool=False,
                            consecutive_ids:bool=False,
                            concurrency:int=4,
                            write_property:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the louvain algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/louvain/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            relationship_weight_property (str, optional): The relationship weight property. Defaults to None.
            seed_property (str, optional): The seed property. Defaults to None.
            max_levels (int, optional): The max levels. Defaults to 10.
            tolerance (float, optional): The tolerance. Defaults to 0.0001.
            include_intermediate_communities (bool, optional): The include intermediate communities. Defaults to False.
            consecutive_ids (bool, optional): The consecutive ids. Defaults to False.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """

        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        if relationship_weight_property:
            algorithm_config['relationshipWeightProperty']=relationship_weight_property
        algorithm_config['seedProperty']=seed_property
        algorithm_config['maxLevels']=max_levels
        algorithm_config['tolerance']=tolerance
        algorithm_config['includeIntermediateCommunities']=include_intermediate_communities
        algorithm_config['consecutiveIds']=consecutive_ids
        algorithm_config['concurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='louvain',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
    
    def run_modularity_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            relationship_weight_property:str=None,
                            community_property:str=None,
                            concurrency:int=4):
        """
        Runs the modularity algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/modularity/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            relationship_weight_property (str, optional): The relationship weight property. Defaults to None.
            community_property (str, optional): The community property. Defaults to None.
            concurrency (int, optional): The concurrency. Defaults to 4.

        Returns:    
            list: Returns results of the algorithm.
        """

        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if community_property is None:
            raise Exception("Community property must be provided")
        
        algorithm_config={}
        if relationship_weight_property:
            algorithm_config['relationshipWeightProperty']=relationship_weight_property
        if community_property:
            algorithm_config['communityProperty']=community_property
        algorithm_config['concurrency']=concurrency

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='modularity',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results

    def run_modularity_optimization_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            max_iterations:int=10,
                            tolerance:float=0.0001,
                            seed_property:str=None,
                            consecutive_ids:bool=False,
                            relationship_weight_property:str=None,
                            min_community_size:int=0,
                            concurrency:int=4,
                            write_property:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the modularity optimization algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/modularity-optimization/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            max_iterations (int, optional): The max iterations. Defaults to 10.
            tolerance (float, optional): The tolerance. Defaults to 0.0001.
            seed_property (str, optional): The seed property. Defaults to None.
            consecutive_ids (bool, optional): The consecutive ids. Defaults to False.
            relationship_weight_property (str, optional): The relationship weight property. Defaults to None.
            min_community_size (int, optional): The min community size. Defaults to 0.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """

        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        algorithm_config['maxIterations']=max_iterations
        algorithm_config['tolerance']=tolerance
        algorithm_config['seedProperty']=seed_property
        algorithm_config['consecutiveIds']=consecutive_ids
        if relationship_weight_property:
            algorithm_config['relationshipWeightProperty']=relationship_weight_property
        algorithm_config['minCommunitySize']=min_community_size
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='modularityOptimization',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
    
    def run_strongly_connected_components_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            consecutive_ids:bool=False,
                            concurrency:int=4,
                            write_property:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the strongly connected components algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/strongly-connected-components/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            consecutive_ids (bool, optional): The consecutive ids. Defaults to False.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """

        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        algorithm_config['consecutiveIds']=consecutive_ids
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property
        
        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='scc',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results

    def run_triangle_count_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            max_degree:int=None,
                            concurrency:int=4,
                            write_property:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the triangle count algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/triangle-count/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            max_degree (int, optional): The max degree. Defaults to None.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """

        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        if max_degree:
            algorithm_config['maxDegree']=max_degree
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='triangleCount',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results

    def run_weakly_connected_components_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            relationship_weight_property:str=None,
                            seed_property:str=None,
                            threshold:float=0.001,
                            consecutive_ids:bool=False,
                            min_component_size:int=0,
                            concurrency:int=4,
                            write_property:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the weakly connected components algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/weakly-connected-components/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            relationship_weight_property (str, optional): The relationship weight property. Defaults to None.
            seed_property (str, optional): The seed property. Defaults to None.
            threshold (float, optional): The threshold. Defaults to 0.001.
            consecutive_ids (bool, optional): The consecutive ids. Defaults to False.
            min_component_size (int, optional): The min component size. Defaults to 0.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """

        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        if relationship_weight_property:
            algorithm_config['relationshipWeightProperty']=relationship_weight_property
        algorithm_config['seedProperty']=seed_property
        algorithm_config['threshold']=threshold
        algorithm_config['consecutiveIds']=consecutive_ids
        algorithm_config['minComponentSize']=min_component_size
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='weaklyConnectedComponents',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
    
    def run_approximate_max_k_cut_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            k:int=2,
                            iterations:int=8,
                            vns_max_neighborhood_order:int=None,
                            random_seed:int=42,
                            relationship_weight_property:str=None,
                            concurrency:int=4,
                            write_property:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the approximate max k cut algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/approximate-max-k-cut/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            k (int, optional): The k. Defaults to 2.
            iterations (int, optional): The iterations. Defaults to 8.
            vns_max_neighborhood_order (int, optional): The vns max neighborhood order. Defaults to None.
            random_seed (int, optional): The random seed. Defaults to 42.
            relationship_weight_property (str, optional): The relationship weight property. Defaults to None.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """

        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        algorithm_config['k']=k
        algorithm_config['iterations']=iterations
        if vns_max_neighborhood_order:
            algorithm_config['vnsMaxNeighborhoodOrder']=vns_max_neighborhood_order
        algorithm_config['randomSeed']=random_seed
        if relationship_weight_property:
            algorithm_config['relationshipWeightProperty']=relationship_weight_property
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='maxkcut',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
                            
    def run_speaker_listener_label_propagation_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            max_iterations:int=None,
                            min_assocation_strength:float=0.2,
                            partitioning:str="RANGE",
                            concurrency:int=4,
                            write_property:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the speaker listener label propagation algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/speaker-listener-label-propagation/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            max_iterations (int, optional): The max iterations. Defaults to None.
            min_assocation_strength (float, optional): The min assocation strength. Defaults to 0.2.
            partitioning (str, optional): The partitioning. Defaults to "RANGE".
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """

        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        if max_iterations is None:
            raise Exception("Max iterations must be provided")
        
        algorithm_config={}
        algorithm_config['maxIterations']=max_iterations
        algorithm_config['minAssocationStrength']=min_assocation_strength
        algorithm_config['partitioning']=partitioning
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='sllpa',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results

    def run_article_ranking_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            damping_factor:float=0.85,
                            max_iterations:int=20,
                            tolerance:float=0.0000001,
                            relationship_weight_property:str=None,
                            source_nodes:List=None,
                            scaler:Union[str,dict]=None,
                            concurrency:int=4,
                            write_property:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the article ranking algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/article-ranking/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            damping_factor (float, optional): The damping factor. Defaults to 0.85.
            max_iterations (int, optional): The max iterations. Defaults to 20.
            tolerance (float, optional): The tolerance. Defaults to 0.0000001.
            relationship_weight_property (str, optional): The relationship weight property. Defaults to None.
            source_nodes (List, optional): The source nodes. Defaults to None.
            scaler (Union[str,dict], optional): The scaler. Defaults to None.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """

        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        algorithm_config['dampingFactor']=damping_factor
        algorithm_config['maxIterations']=max_iterations
        algorithm_config['tolerance']=tolerance
        if relationship_weight_property:
            algorithm_config['relationshipWeightProperty']=relationship_weight_property
        if source_nodes:
            algorithm_config['sourceNodes']=source_nodes
        if scaler:
            algorithm_config['scaler']=scaler
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='articleRank',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
    
    def run_betweenness_centrality_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            sampling_size:int=None,
                            sampling_seed:int=None,
                            relationship_weight_property:str=None,
                            concurrency:int=4,
                            write_property:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the betweenness centrality algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/betweenness-centrality/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            sampling_size (int, optional): The sampling size. Defaults to None.
            sampling_seed (int, optional): The sampling seed. Defaults to None.
            relationship_weight_property (str, optional): The relationship weight property. Defaults to None.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """

        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        if sampling_size:
            algorithm_config['samplingSize']=sampling_size
        if sampling_seed:
            algorithm_config['samplingSeed']=sampling_seed
        if relationship_weight_property:
            algorithm_config['relationshipWeightProperty']=relationship_weight_property
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='betweenness',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
    
    def run_celf_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            seed_set_size:int=None,
                            monte_carlo_simulations:int=100,
                            propagation_probability:float=0.1,
                            random_seed:int=None,
                            concurrency:int=4,):
        """
        Runs the celf algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/celf/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            seed_set_size (int, optional): The seed set size. Defaults to None.
            monte_carlo_simulations (int, optional): The monte carlo simulations. Defaults to 100.
            propagation_probability (float, optional): The propagation probability. Defaults to 0.1.
            random_seed (int, optional): The random seed. Defaults to None.
            concurrency (int, optional): The concurrency. Defaults to 4.
        
        Returns:    
            list: Returns results of the algorithm.
        """

        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        
        algorithm_config={}
        if seed_set_size:
            algorithm_config['seedSetSize']=seed_set_size
        if monte_carlo_simulations:
            algorithm_config['monteCarloSimulations']=monte_carlo_simulations
        if propagation_probability:
            algorithm_config['propagationProbability']=propagation_probability
        if random_seed:
            algorithm_config['randomSeed']=random_seed
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='influenceMaximization',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
    
    def run_closeness_centrality_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            use_wasserman_faust:bool=False,
                            concurrency:int=4,
                            write_property:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the closeness centrality algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/closeness-centrality/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            use_wasserman_faust (bool, optional): The use wasserman faust. Defaults to False.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """

        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        
        algorithm_config={}

        algorithm_config['useWassermanFaust']=use_wasserman_faust
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='closeness',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
    
    def run_degree_centrality_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            orientation:str='NATURAL',
                            relationship_weight_property:str=None,
                            concurrency:int=4,
                            write_property:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the degree centrality algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/degree-centrality/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            orientation (str, optional): The orientation. Defaults to 'NATURAL'.
            relationship_weight_property (str, optional): The relationship weight property. Defaults to None.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """

        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        algorithm_config['orientation']=orientation
        algorithm_config['concurrency']=concurrency
        if relationship_weight_property:
            algorithm_config['relationshipWeightProperty']=relationship_weight_property
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='degree',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results

    def run_eigenvector_centrality_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            max_iterations:int=20,
                            tolerance:float=0.0000001,
                            relationship_weight_property:str=None,
                            source_nodes:List=None,
                            scaler:Union[str,dict]=None,
                            concurrency:int=4,
                            write_property:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the eigenvector centrality algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/eigenvector-centrality/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            max_iterations (int, optional): The max iterations. Defaults to 20.
            tolerance (float, optional): The tolerance. Defaults to 0.0000001.
            relationship_weight_property (str, optional): The relationship weight property. Defaults to None.
            source_nodes (List, optional): The source nodes. Defaults to None.
            scaler (Union[str,dict], optional): The scaler. Defaults to None.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """
        
        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        algorithm_config['maxIterations']=max_iterations
        algorithm_config['tolerance']=tolerance
        if relationship_weight_property:
            algorithm_config['relationshipWeightProperty']=relationship_weight_property
        if source_nodes:
            algorithm_config['sourceNodes']=source_nodes
        if scaler:
            algorithm_config['scaler']=scaler
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='eigenvector',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
    
    def run_page_rank_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            damping_factor:float=0.85,
                            max_iterations:int=20,
                            tolerance:float=0.0000001,
                            relationship_weight_property:str=None,
                            source_nodes:List=None,
                            scaler:Union[str,dict]=None,
                            concurrency:int=4,
                            write_property:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the page rank algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/page-rank/

        Args:    
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            damping_factor (float, optional): The damping factor. Defaults to 0.85.
            max_iterations (int, optional): The max iterations. Defaults to 20.
            tolerance (float, optional): The tolerance. Defaults to 0.0000001.
            relationship_weight_property (str, optional): The relationship weight property. Defaults to None.
            source_nodes (List, optional): The source nodes. Defaults to None.
            scaler (Union[str,dict], optional): The scaler. Defaults to None.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """
        
        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:    
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        algorithm_config['dampingFactor']=damping_factor
        algorithm_config['maxIterations']=max_iterations
        algorithm_config['tolerance']=tolerance
        if relationship_weight_property:
            algorithm_config['relationshipWeightProperty']=relationship_weight_property
        if source_nodes:
            algorithm_config['sourceNodes']=source_nodes
        if scaler:
            algorithm_config['scaler']=scaler
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='pageRank',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
    
    def run_harmonic_centrality_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            concurrency:int=4,
                            write_property:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the harmonic centrality algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/harmonic-centrality/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """
        
        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='closeness.harmonic',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results
                            
    def run_hits_centrality_algorithm(self,
                            database_name:str,
                            graph_name:str,
                            algorithm_mode:str,
                            hits_iterations:int=20,
                            auth_property:str=None,
                            hub_property:str=None,
                            partitioning:str='AUTO',
                            concurrency:int=4,
                            write_property:str=None,
                            mutate_property:str=None,
                            ):
        """
        Runs the hits centrality algorithm.
        Useful urls:
            https://neo4j.com/docs/graph-data-science/current/machine-learning/hits-centrality/

        Args:
            database_name (str): The name of the database.
            graph_name (str): The name of the graph.
            algorithm_mode (str, optional): The mode of the algorithm. Defaults to 'stream'.
            hits_iterations (int, optional): The hits iterations. Defaults to 20.
            auth_property (str, optional): The auth property. Defaults to None.
            hub_property (str, optional): The hub property. Defaults to None.
            partitioning (str, optional): The partitioning. Defaults to 'AUTO'.
            concurrency (int, optional): The concurrency. Defaults to 4.
            write_property (str, optional): The write property. Defaults to None.
            mutate_property (str, optional): The mutate property. Defaults to None.

        Returns:    
            list: Returns results of the algorithm.
        """
        
        if algorithm_mode not in self.algorithm_modes:
            raise Exception(f"Algorithm mode {algorithm_mode} is not supported. Use one of {self.algorithm_modes}")
        if algorithm_mode=='write' and write_property is None:
            raise Exception("write_property must be provided when algorithm_mode is write")
        if algorithm_mode=='mutate' and mutate_property is None:
            raise Exception("mutate_property must be provided when algorithm_mode is mutate")
        
        algorithm_config={}
        algorithm_config['hitsIterations']=hits_iterations
        algorithm_config['authProperty']=auth_property
        algorithm_config['hubProperty']=hub_property
        algorithm_config['partitioning']=partitioning
        algorithm_config['concurrency']=concurrency
        if algorithm_mode=='write':
            algorithm_config['writeConcurrency']=concurrency
        if write_property:
            algorithm_config['writeProperty']=write_property
        if mutate_property:
            algorithm_config['mutateProperty']=mutate_property

        results=self.run_algorithm(self,
                        database_name=database_name,
                        graph_name=graph_name,
                        algorithm_name='hits',
                        algorithm_mode=algorithm_mode,
                        algorithm_config=algorithm_config)
        return results

if __name__ == "__main__":

    from matgraphdb.graph_kit.neo4j.neo4j_manager import Neo4jManager

    # from matgraphdb.graph.graph_generator import GraphGenerator
    # generator=GraphGenerator(from_scratch=False)
    # main_graph_dir=generator.main_graph_dir
    # sub_graph_names=generator.list_sub_graphs()
    # print(sub_graph_names)
    # sub_graph_paths=[os.path.join(main_graph_dir,'sub_graphs',sub_graph_name) for sub_graph_name in sub_graph_names]
    
    # with Neo4jGraphDatabase(from_scratch=True) as manager:
    #     for path in sub_graph_paths:
    #         # print(manager.list_databases())
    #         results=manager.load_graph_database_into_neo4j(path)
    #         print(results)


    # with Neo4jGraphDatabase() as matgraphdb:
        # database_name='nelements-no-fe'
    #     graph_dir=os.path.join('data','production','materials_project','graph_database')
    #     # settings={'apoc.export.file.enabled':'true'}
    #     # matgraphdb.set_apoc_environment_variables(settings=settings)
    #     matgraphdb.export_database(graph_dir,database_name=database_name)
    # with Neo4jGraphDatabase() as matgraphdb:
    # #     database_name='nelements-1-2'
    #     manager=Neo4jGDSManager(matgraphdb)
    # #     print(manager.list_graphs(database_name))
    # #     print(manager.is_graph_in_memory(database_name,'materials_chemenvElements'))
    # #     # print(manager.list_graph_data_science_algorithms(database_name))
    #     database_name='elements-no-fe'
    #     graph_name='materials_chemenvElements'
    # #     node_projections=['ChemenvElement','Material']
    # #     relationship_projections={
    # #                 "GEOMETRIC_ELECTRIC_CONNECTS": {
    # #                 "orientation": 'UNDIRECTED',
    # #                 "properties": 'weight'
    # #                 },
    # #                 "COMPOSED_OF": {
    # #                     "orientation": 'UNDIRECTED',
    # #                     "properties": 'weight'
    # #                 }
    # #             }
        
    # #     # print(format_dictionary(relationship_projections))
    # #     manager.load_graph_into_memory(database_name=database_name,
    # #                                    graph_name=graph_name,
    # #                                    node_projections=node_projections,
    # #                                    relationship_projections=relationship_projections)
    #     # print(manager.get_graph_info(database_name=database_name,graph_name=graph_name))
    #     print(manager.list_graphs(database_name))
        # print(manager.is_graph_in_memory(database_name,graph_name))
        # print(manager.drop_graph(database_name,graph_name))
        # print(manager.list_graphs(database_name))
        # print(manager.is_graph_in_memory(database_name,graph_name))

        # print(manager.get_graph_info(database_name=database_name,graph_name=graph_name))
        # results=manager.read_graph(database_name=database_name,
        #                    graph_name=graph_name,
        #                    node_properties=["fastrp-embedding"],
        #                    node_labels=['Material'])
        
        # print(results)


        # result=manager.estimate_memeory_for_algorithm(database_name=database_name,
        #                                        graph_name=graph_name,
        #                                        algorithm_name='fastRP',
        #                                        model='stream',
        #                                        algorithm_config={'embeddingDimension':128})
        # result=manager.run_algorithm(database_name=database_name,
        #                                        graph_name=graph_name,
        #                                        algorithm_name='fastRP',
        #                                        algorithm_mode='stats',
        #                                        algorithm_config={'embeddingDimension':128})
        # print(result)






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




    ##################################################################################################
    # # Loading and unloading graphs into GDS
    ##################################################################################################
    with Neo4jManager() as matgraphdb:
        database_name='elements-no-fe'
        graph_name='materials_chemenvElements'
        manager=Neo4jGDSManager(matgraphdb)
        print(manager.list_graphs(database_name))
        print(manager.is_graph_in_memory(database_name,'materials_chemenvElements'))
        print(manager.drop_graph(database_name,graph_name))
    #     graph_name='materials_chemenvElements'
    #     node_projections={
    #             "ChemenvElement":{
    #                 "label":'ChemenvElement',
    #                 },
    #             "Material":{
    #                 "label":'Material',
    #                 "properties":['band_gap','formation_energy_per_atom','energy_per_atom','energy_above_hull','k_vrh','g_vrh']
    #             }
    #     }
    #     relationship_projections = {
    #         "GEOMETRIC_ELECTRIC_CONNECTS": {
    #             "orientation": 'UNDIRECTED',
    #             "properties": 'weight'
    #         },
    #         "COMPOSED_OF": {
    #             "orientation": 'UNDIRECTED',
    #             "properties": 'weight'
    #         }
    #     }
        # manager.load_graph_into_memory(database_name=database_name,
        #                                graph_name=graph_name,
        #                                node_projections=node_projections,
        #                                relationship_projections=relationship_projections)

    ##################################################################################################
    # # Testing Neo4jDLManager
    ##################################################################################################
    # with MatGraphDB() as matgraphdb:
    #     database_name='nelements-1-2'
    #     manager=Neo4jDLManager(matgraphdb)
    #     print(manager.list_graphs(database_name))
    #     print(manager.is_graph_in_memory(database_name,'materials_chemenvElements'))
        # results=manager.list_graph_data_science_algorithms(database_name,save=True)
        # for result in results:
        #     # print the reuslts in two columns
        #     print(result[0],'|||||||||||||||',result[1])