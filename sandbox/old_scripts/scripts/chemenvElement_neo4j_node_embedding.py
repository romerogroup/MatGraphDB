

import os
import ast
import json
import copy
from typing import List

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from matgraphdb.mlcore.trainer import Trainer
from matgraphdb.mlcore.datasets import NumpyDataset
from matgraphdb.mlcore.models import MultiLayerPerceptron, WeightedRandomClassifier, MajorityClassClassifier,LinearRegressor
from matgraphdb import Neo4jManager,Neo4jGDSManager,GraphGenerator
from matgraphdb.utils import ML_SCRATCH_RUNS_DIR

import time
# torch.manual_seed(0)
# np.random.seed(0)

def load_graphs_into_neo4j(from_scratch=False):
    generator=GraphGenerator(from_scratch=False)
    main_graph_dir=generator.main_graph_dir
    sub_graph_names=generator.list_sub_graphs()
    print(sub_graph_names)
    sub_graph_paths=[os.path.join(main_graph_dir,'sub_graphs',sub_graph_name) for sub_graph_name in sub_graph_names]
    for path in sub_graph_paths:
        load_graph_into_neo4j(graph_path=path, from_scratch=False)

def load_graph_into_neo4j(graph_path, graph_name=None, from_scratch=False):
    with Neo4jManager(from_scratch=True) as manager:
        results=manager.load_graph_database_into_neo4j(database_path=graph_path,new_database_name=graph_name)

def train_node_embedding_pipeline(database_name,
                            graph_name, 
                            fastrp_params,
                            node_projections,
                            relationship_projections):
    with Neo4jManager() as neo4j_manager:
        manager=Neo4jGDSManager(neo4j_manager)
        
        print(manager.is_graph_in_memory(database_name,graph_name))
        try:
            manager.load_graph_into_memory(database_name=database_name,
                                            graph_name=graph_name,
                                            node_projections=node_projections,
                                            relationship_projections=relationship_projections)
        
            print(manager.is_graph_in_memory(database_name,graph_name))

            results=manager.run_fastRP_algorithm(database_name=database_name,
                                        graph_name=graph_name,
                                        **fastrp_params)
            manager.drop_graph(database_name=database_name,graph_name=graph_name)
            print(manager.is_graph_in_memory(database_name,graph_name))
            return results
        except Exception as e:
            print(f"Database {database_name} : {e}")
            pass

def train_all_graph_node_embeddings(node_projections,relationship_projections,write_property,from_scratch=False):
    with Neo4jManager() as neo4j_manager:
        db_names=neo4j_manager.list_databases()

    graph_name = 'materials_chemenvElements'
    
    # Define algorithm parameters
    parameters = {
        "algorithm_mode": 'write',
        'embedding_dimension':128,
        "write_property": write_property
    }
    
    for database_name in db_names:
        print('-'*100)
        print(database_name)
        with Neo4jManager() as neo4j_manager:
            property_exists=neo4j_manager.does_property_exist(database_name,'Material',parameters['write_property'])
        if property_exists and not from_scratch:
            continue

        train_node_embedding_pipeline(database_name,
                                    graph_name,
                                    fastrp_params=parameters,
                                    node_projections=node_projections,
                                    relationship_projections=relationship_projections)

# def train_experiment_node_embeddings(database_names=None,from_scratch=False, fast_rp_params={}):
#     gec_element_chemenv_relationship_projections = {
#         "GEOMETRIC_ELECTRIC_CONNECTS": {
#             "orientation": 'UNDIRECTED',
#             "properties": 'weight'
#         },
#         "COMPOSED_OF": {
#             "orientation": 'UNDIRECTED',
#             "properties": 'weight'
#         },
#         "CAN_OCCUR": {
#             "orientation": 'UNDIRECTED',
#             "properties": 'weight'
#         }
#     }
#     gc_element_chemenv_relationship_projections = {
#         "GEOMETRIC_CONNECTS": {
#             "orientation": 'UNDIRECTED',
#             "properties": 'weight'
#         },
#         "COMPOSED_OF": {
#             "orientation": 'UNDIRECTED',
#             "properties": 'weight'
#         },
#         "CAN_OCCUR": {
#             "orientation": 'UNDIRECTED',
#             "properties": 'weight'
#         }
#     }
#     ec_element_chemenv_relationship_projections = {
#         "ELECTRIC_CONNECTS": {
#             "orientation": 'UNDIRECTED',
#             "properties": 'weight'
#         },
#         "COMPOSED_OF": {
#             "orientation": 'UNDIRECTED',
#             "properties": 'weight'
#         },
#         "CAN_OCCUR": {
#             "orientation": 'UNDIRECTED',
#             "properties": 'weight'
#         }
#     }

#     experiment_list=[(['Chemenv', 'Element', 'Material'],gec_element_chemenv_relationship_projections,'fastrp-embedding-gec-element-chemenv'),
#                     (['Chemenv', 'Element', 'Material'],gc_element_chemenv_relationship_projections,'fastrp-embedding-gc-element-chemenv'),
#                     (['Chemenv', 'Element', 'Material'],ec_element_chemenv_relationship_projections,'fastrp-embedding-ec-element-chemenv')]
#     with Neo4jManager() as neo4j_manager:
#         db_names=neo4j_manager.list_databases()

#     graph_name = 'materials_chemenvElements'
#     if database_names:
#         db_names=database_names
#     for database_name in db_names:
#         print('-'*100)
#         print(database_name)
        

#         for node_projections,relationship_projections,write_property in experiment_list:
#             print(write_property)
#             # Define algorithm parameters
#             fast_rp_params['write_property']=write_property
            
#             with Neo4jManager() as neo4j_manager:
#                 property_exists=neo4j_manager.does_property_exist(database_name,'Material',fast_rp_params['write_property'])
#             if property_exists and not from_scratch:
#                 continue

#             train_node_embedding_pipeline(database_name,
#                                     graph_name,
#                                     fastrp_params=fast_rp_params,
#                                     node_projections=node_projections,
#                                     relationship_projections=relationship_projections)
            
def train_experiment_node_embeddings(database_names=None,from_scratch=False, fast_rp_params={}):
    gec_element_chemenv_relationship_projections = {
        "`CHEMENV-ELECTRIC_CONNECTS-CHEMENV`": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "`ELEMENT-ELECTRIC_CONNECTS-ELEMENT`": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "`CHEMENV-CAN_OCCUR-ELEMENT`": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "`MATERIAL-HAS-ELEMENT`": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "`MATERIAL-HAS-CHEMENV`": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        }
    }
    gc_element_chemenv_relationship_projections = {
        "`CHEMENV-GEOMETRIC_CONNECTS-CHEMENV`": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "`ELEMENT-GEOMETRIC_CONNECTS-ELEMENT`": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "`CHEMENV-CAN_OCCUR-ELEMENT`": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "`MATERIAL-HAS-ELEMENT`": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "`MATERIAL-HAS-CHEMENV`": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        }
    }
    ec_element_chemenv_relationship_projections = {
        "`CHEMENV-GEOMETRIC_ELECTRIC_CONNECTS-CHEMENV`": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "`ELEMENT-GEOMETRIC_ELECTRIC_CONNECTS-ELEMENT`": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "`CHEMENV-CAN_OCCUR-ELEMENT`": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "`MATERIAL-HAS-ELEMENT`": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "`MATERIAL-HAS-CHEMENV`": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        }
    }

    experiment_list=[(['Chemenv', 'Element', 'Material'],gec_element_chemenv_relationship_projections,'fastrp-embedding-gec-element-chemenv'),
                    (['Chemenv', 'Element', 'Material'],gc_element_chemenv_relationship_projections,'fastrp-embedding-gc-element-chemenv'),
                    (['Chemenv', 'Element', 'Material'],ec_element_chemenv_relationship_projections,'fastrp-embedding-ec-element-chemenv'),
                    (['Chemenv', 'Element', 'Material'],gec_element_chemenv_relationship_projections,'fastrp-embedding-gec-element-chemenv-weighted'),
                    (['Chemenv', 'Element', 'Material'],gc_element_chemenv_relationship_projections,'fastrp-embedding-gc-element-chemenv-weighted'),
                    (['Chemenv', 'Element', 'Material'],ec_element_chemenv_relationship_projections,'fastrp-embedding-ec-element-chemenv-weighted')]
    with Neo4jManager() as neo4j_manager:
        db_names=neo4j_manager.list_databases()

    graph_name = 'materials_chemenvElements'
    if database_names:
        db_names=database_names
    for database_name in db_names:
        print('-'*100)
        print(database_name)
        

        for node_projections,relationship_projections,write_property in experiment_list:
            print(write_property)
            # Define algorithm parameters
            fast_rp_params['write_property']=write_property
            
            with Neo4jManager() as neo4j_manager:
                property_exists=neo4j_manager.does_property_exist(database_name,'Material',fast_rp_params['write_property'])
            if property_exists and not from_scratch:
                continue

            train_node_embedding_pipeline(database_name,
                                    graph_name,
                                    fastrp_params=fast_rp_params,
                                    node_projections=node_projections,
                                    relationship_projections=relationship_projections)

def get_embeddings_and_properties_df(database_name:str,node_type:str,embedding_properties:List[str], node_properties:List[str]):
    with Neo4jManager() as neo4j_manager:
        
        cypher_statement=f"""MATCH (m:{node_type}) RETURN """
        cypher_statement+=f"""m.name AS name,"""
        n_emb_properties=len(embedding_properties)
        for i,embedding_property in enumerate(embedding_properties):
            cypher_statement+=f"""m.`{embedding_property}` AS `{embedding_property}`"""
            cypher_statement+=f", "

        n_properties=len(node_properties)
        for i,node_property in enumerate(node_properties):
            cypher_statement+=f"""m.`{node_property}` AS `{node_property}`"""
            if i!=n_properties-1:
                cypher_statement+=f", "

        results=neo4j_manager.query(cypher_statement,database_name=database_name)

    emb_df = pd.DataFrame([dict(_) for _ in results])
    emb_df = emb_df.replace({np.nan: None})
    properties_df=emb_df[node_properties]
    property_types=get_property_types(properties_df)
    return emb_df,property_types

def get_property_types(properties_df):
    properties=list(properties_df.columns)
    property_types={}
    for property in properties[:]:
        for irow,row in properties_df.iterrows():
            if row[property] is None:
                continue
            property_type=type(row[property])
            property_types[property]=property_type
            break
    return property_types


class EmbeddingBenchmark:

    def __init__(self,database_name='nelements-2-2',node_type='Material',metrics_path='.'):
        self.embedding_properties=['fastrp-embedding-gc-element-chemenv-weighted',
                                   'fastrp-embedding-ec-element-chemenv-weighted',
                                   'fastrp-embedding-gec-element-chemenv-weighted']
        # self.embedding_properties=['fastrp-embedding-gec-element-chemenv-weighted']
        # self.properties=['nsites','nelements','volume','density','density_atomic','formula_pretty',
        #                 'e_electronic','e_ionic','e_total','energy_per_atom','energy_above_hull','formation_energy_per_atom',
        #                 'band_gap','vbm','cbm','efermi',
        #                 'crystal_system','space_group','point_group','hall_symbol',
        #                 'is_gap_direct','is_metal','is_magnetic','is_stable',
        #                 'ordering','total_magnetization','total_magnetization_normalized_vol','num_magnetic_sites','num_unique_magnetic_sites',
        #                 'g_reuss','g_voigt','g_vrh','k_reuss','k_voigt','k_vrh','homogeneous_poisson','universal_anisotropy']
        
        self.properties=['nsites','nelements','volume','density','density_atomic','crystal_system','space_group']
        self.properties=['crystal_system','nelements']
        
        df,property_types=get_embeddings_and_properties_df(database_name=database_name,
                                                            node_type=node_type,
                                                            embedding_properties=self.embedding_properties, 
                                                            node_properties=self.properties)
        self.df=df
        self.property_types_map=property_types
        self.metrics_dict={prop_key: {embedding_key:{} for embedding_key in self.embedding_properties} for prop_key in self.properties}
        self.metrics_path=os.path.join(metrics_path,database_name+'.json')
        self.device=  "cuda:0" if torch.cuda.is_available() else torch.device("cpu")
        os.makedirs(metrics_path,exist_ok=True)

        self.ml_tasks={}

        self._get_ml_tasks()

    def _get_ml_tasks(self):
        for key,val in self.property_types_map.items():
            if key=='space_group':
                self.ml_tasks[key]='classification'
            elif val==str:
                self.ml_tasks[key]='classification'
            elif val==bool:
                self.ml_tasks[key]='classification'
            elif val==int:
                self.ml_tasks[key]='regression'
            elif val==float:
                self.ml_tasks[key]='regression'
        return self.ml_tasks

    def _prepare_data(self,embedding_name, property_name):
        property_type=self.property_types_map[property_name]
        property_df=self.df[[property_name,embedding_name]].dropna()
        
        ml_task=self.ml_tasks[property_name]
        class_names=None
        id_class_map=None
        class_id_map=None
        if ml_task=='classification':
            # sort the class names
            class_names=sorted(property_df[property_name].unique())
            class_id_map={class_name:i for i,class_name in enumerate(class_names)}
            id_class_map={i:class_name for i,class_name in enumerate(class_names)}

            # Map the class names to IDs
            property_df[property_name] = property_df[property_name].map(class_id_map)

            property_type=np.int64
        else:
            property_type=np.float32

        property_df['X'] = property_df[embedding_name].apply(lambda x: np.array(x,np.float32))
        X = np.array(property_df['X'].to_list(),np.float32)

        y=property_df[property_name].values.astype(property_type)
        return X, y, id_class_map, ml_task
    
    def _create_torch_dataset(self,X_train, X_test, y_train, y_test):
        train_dataset=NumpyDataset(X_train,y_train)
        test_dataset=NumpyDataset(X_test,y_test)
        return train_dataset, test_dataset
 
    def _split_data(self,X,y,test_size=0.2,random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    
    def _get_class_counts(self,dataset,number_of_classes):
        
        train_loader=DataLoader(dataset=dataset, batch_size=128, shuffle=True)
        class_counts = torch.zeros(number_of_classes)  # Replace `number_of_classes` with your actual number of classes
        for _,labels in train_loader:
            class_counts += torch.bincount(labels, minlength=number_of_classes)

        # Prevent division by zero in case some class is not present at all
        class_weights = 1. / (class_counts + 1e-5)  
        # Normalize weights so that the smallest weight is 1.0
        class_weights = class_weights / class_weights.min()
        train_loader=None
        return class_counts,class_weights
    
    def _get_majority_class(self,class_weights):
        majority_class=torch.argmax(class_weights)
        return majority_class
    
    def _train_model(self,
                     train_dataset, 
                     test_dataset, 
                     ml_task, 
                     id_class_map, 
                     train_params, 
                     run_path,
                    **kwargs):

        
        if ml_task=='classification':
            output_dim=len(id_class_map)
            class_counts,class_weights=self._get_class_counts(dataset=train_dataset,number_of_classes=output_dim)
            majority_class=self._get_majority_class(class_counts)
            loss_fn=nn.CrossEntropyLoss()
            models=[MultiLayerPerceptron,MajorityClassClassifier,WeightedRandomClassifier]
            model_names=['MLP','Majority','Weighted Random']
            models_params=[{'input_dim':train_params['input_dim'],
                            'output_dim':output_dim,
                            'num_layers':train_params['num_layers'],
                            'n_embd':train_params['n_embd']},
                            {'majority_class':majority_class,
                            'num_classes':output_dim},
                            {'class_weights':class_counts}]
            
        elif ml_task=='regression':
            output_dim=1
            loss_fn=nn.MSELoss()
            models=[LinearRegressor,MultiLayerPerceptron]
            model_names=['Linear Regressor','MLP']
            models_params=[{'input_dim':train_params['input_dim'],
                            'output_dim':output_dim,
                            'num_layers':train_params['num_layers'],
                            'n_embd':train_params['n_embd']},
                            {'input_dim':train_params['input_dim'],
                            'output_dim':output_dim}]
        
        # define the model
        for uninit_model,model_params,model_name in zip(models,models_params,model_names):
            model = uninit_model(**model_params)
            model.to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=train_params['learning_rate'])

            run_name=f"{model_name}"
            trainer=Trainer(train_dataset, test_dataset, model, loss_fn, optimizer, self.device,
                            run_path=run_path,
                            run_name=run_name,
                            eval_interval=train_params['eval_interval'],
                            max_iters=train_params['max_iters'],
                            early_stopping_patience=train_params['patience'])
            trainer.train()
    
    def train_models(self,train_params):
        for embedding_property in self.embedding_properties[:]:
            for property in self.properties[:1]:
                print("-"*100)
                print(f"Training {property} on {embedding_property}")

                X,y,id_class_map,ml_task=self._prepare_data(embedding_property,property)

                X_train, X_test, y_train, y_test = self._split_data(X,y,test_size=train_params['test_size'],
                                                                    random_state=train_params['random_state'])
                train_dataset, test_dataset=self._create_torch_dataset(X_train, X_test, y_train, y_test)

                
                run_path=os.path.join(ML_SCRATCH_RUNS_DIR,f"{property}-{embedding_property}")
                print(run_path)
                self._train_model(train_dataset,
                                test_dataset,
                                ml_task,
                                id_class_map,
                                train_params=train_params,
                                run_path=run_path)





def main():

    ########################################################################################################
    # Loading graphs into Neo4j
    ########################################################################################################
    # load_graphs_into_neo4j(from_scratch=False)
    # load_graph_into_neo4j(from_scratch=False)


    ########################################################################################################
    # Creating graph node embeddings
    ########################################################################################################
    # fast_rp_params={
    #     'algorithm_mode': 'write',
    #     'embedding_dimension':128,
    #     'relationship_weight_property': 'weight'
    # }
    # train_experiment_node_embeddings(database_names=['main-06242024'],from_scratch=True,fast_rp_params=fast_rp_params)
    # embedding_properties=['fastrp-embedding-gc-chemenvElement','fastrp-embedding-ec-chemenvElement','fastrp-embedding-gec-chemenvElement',
    #                       'fastrp-embedding-gc-element-chemenv','fastrp-embedding-ec-element-chemenv','fastrp-embedding-gec-element-chemenv']
    # node_properties=['nsites','nelements','volume','density','density_atomic','composition_reduced','formula_pretty'
    #                  'e_electronic','e_ionic','e_total','energy_per_atom','energy_above_hull','formation_energy_per_atom',
    #                  'band_gap','vbm','cbm','efermi',
    #                  'crystal_system','space_group','point_group','hall_symbol'
    #                  'is_gap_direct','is_metal','is_magnetic','is_stable',
    #                  'ordering','total_magnetization','total_magnetization_normalized_vol','num_magnetic_sites','num_unique_magnetic_sites'
    #                  'g_ruess','g_voigt','g_vrh','k_reuss','k_voigt','k_vrh','homogeneous_poisson','universal_anisotropy']
    
    ############################################################################################################

    ########################################################################################################
    # Training model on material properties
    ########################################################################################################
    train_params={
        'batch_size':64,
        'random_state':42,
        'test_size':0.1,
        'learning_rate':0.0004,
        'eval_interval':2,
        'max_iters':50,
        'patience':5,
        'num_layers':1,
        'n_embd':64,
        'input_dim':128
    }

    # metrics_path='examples/scripts/base_metrics.json'
    # databases=['nelements-2-3','nelements-3-3','main']
    databases=['main-06242024']
    metrics_path='examples/scripts/testing'
    start_time=time.time()
    for database_name in databases:
        benchmark=EmbeddingBenchmark(database_name=database_name,node_type='Material',metrics_path=metrics_path)
        benchmark.train_models(train_params=train_params)

    end_time=time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    ############################################################################################################
    
    
    
    
    
    
    # embeddings_df, properties_df=get_embeddings_and_properties_df(database_name='nelements-2-2',node_type='Material',embedding_properties=embedding_properties, node_properties=node_properties)
    # print(embeddings_df.head())
    
    # print(properties_df['space_group'].unique())
    # 
    #     
    # print(node_types)
    # with Neo4jManager() as matgraphdb:
    #     manager=Neo4jGDSManager(matgraphdb)
    #     print(manager.get_graph_info(database_name=database_name,graph_name=graph_name))


if __name__=='__main__':
    main()