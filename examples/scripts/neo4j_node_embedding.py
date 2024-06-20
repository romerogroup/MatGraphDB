

import os
import ast
import json
from typing import List

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from matgraphdb import Neo4jManager,Neo4jGDSManager,GraphGenerator

def load_graphs_into_neo4j(from_scratch=False):
    generator=GraphGenerator(from_scratch=False)
    main_graph_dir=generator.main_graph_dir
    sub_graph_names=generator.list_sub_graphs()
    print(sub_graph_names)
    sub_graph_paths=[os.path.join(main_graph_dir,'sub_graphs',sub_graph_name) for sub_graph_name in sub_graph_names]
    with Neo4jManager(from_scratch=True) as manager:
        for path in sub_graph_paths:
            if not from_scratch:
                continue
            results=manager.load_graph_database_into_neo4j(path)
            print(results)

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

def train_experiment_node_embeddings(database_names=None,from_scratch=False):
    gec_relationship_projections = {
        "GEOMETRIC_ELECTRIC_CONNECTS": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "COMPOSED_OF": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        }
    }
    gc_relationship_projections = {
        "GEOMETRIC_CONNECTS": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "COMPOSED_OF": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        }
    }
    ec_relationship_projections = {
        "ELECTRIC_CONNECTS": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "COMPOSED_OF": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        }
    }
    gec_element_chemenv_relationship_projections = {
        "GEOMETRIC_ELECTRIC_CONNECTS": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "COMPOSED_OF": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "CAN_OCCUR": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        }
    }
    gc_element_chemenv_relationship_projections = {
        "GEOMETRIC_CONNECTS": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "COMPOSED_OF": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "CAN_OCCUR": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        }
    }
    ec_element_chemenv_relationship_projections = {
        "ELECTRIC_CONNECTS": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "COMPOSED_OF": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        },
        "CAN_OCCUR": {
            "orientation": 'UNDIRECTED',
            "properties": 'weight'
        }
    }

    experiment_list=[(['ChemenvElement', 'Material'],gec_relationship_projections,'fastrp-embedding-gec-chemenvElement'),
                           (['ChemenvElement', 'Material'],gc_relationship_projections,'fastrp-embedding-gc-chemenvElement'),
                           (['ChemenvElement', 'Material'],ec_relationship_projections,'fastrp-embedding-ec-chemenvElement'),
                           (['Chemenv', 'Element', 'Material'],gec_element_chemenv_relationship_projections,'fastrp-embedding-gec-element-chemenv'),
                           (['Chemenv', 'Element', 'Material'],gc_element_chemenv_relationship_projections,'fastrp-embedding-gc-element-chemenv'),
                           (['Chemenv', 'Element', 'Material'],ec_element_chemenv_relationship_projections,'fastrp-embedding-ec-element-chemenv')]
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
            parameters = {
                "algorithm_mode": 'write',
                'embedding_dimension':128,
                "write_property": write_property
            }
            with Neo4jManager() as neo4j_manager:
                property_exists=neo4j_manager.does_property_exist(database_name,'Material',parameters['write_property'])
            if property_exists and not from_scratch:
                continue

            train_node_embedding_pipeline(database_name,
                                    graph_name,
                                    fastrp_params=parameters,
                                    node_projections=node_projections,
                                    relationship_projections=relationship_projections)

def get_embeddings_and_properties_df(database_name:str,node_type:str,embedding_properties:List[str], node_properties:List[str]):
    with Neo4jManager() as neo4j_manager:
        
        cypher_statement=f"""MATCH (m:{node_type}) RETURN """
        cypher_statement+=f"""m.name AS name,"""
        n_properties=len(node_properties)
        for i,embedding_property in enumerate(embedding_properties):
            cypher_statement+=f"""m.`{embedding_property}` AS `{embedding_property}`"""
            if i!=n_properties-1:
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
        self.embedding_properties=['fastrp-embedding-gc-chemenvElement','fastrp-embedding-ec-chemenvElement','fastrp-embedding-gec-chemenvElement',
                          'fastrp-embedding-gc-element-chemenv','fastrp-embedding-ec-element-chemenv','fastrp-embedding-gec-element-chemenv']
        self.properties=['nsites','nelements','volume','density','density_atomic','composition_reduced','formula_pretty'
                        'e_electronic','e_ionic','e_total','energy_per_atom','energy_above_hull','formation_energy_per_atom',
                        'band_gap','vbm','cbm','efermi',
                        'crystal_system','space_group','point_group','hall_symbol'
                        'is_gap_direct','is_metal','is_magnetic','is_stable',
                        'ordering','total_magnetization','total_magnetization_normalized_vol','num_magnetic_sites','num_unique_magnetic_sites'
                        'g_ruess','g_voigt','g_vrh','k_reuss','k_voigt','k_vrh','homogeneous_poisson','universal_anisotropy']
        df,property_types=get_embeddings_and_properties_df(database_name='nelements-2-2',
                                                                      node_type='Material',
                                                                      embedding_properties=self.embedding_properties, 
                                                                      node_properties=self.properties)
        self.df=df
        self.property_types_map=property_types
        self.metrics_dict={prop_key: {embedding_key:{} for embedding_key in self.embedding_properties} for prop_key in self.properties}
        self.metrics_path=metrics_path
        self.ml_tasks={}
        self._get_ml_tasks()

    def _get_ml_tasks(self):
        for key,val in self.property_types_map.items():
            if val==str:
                self.ml_tasks[key]='classification'
            elif val==bool:
                self.ml_tasks[key]='classification'
            elif val==int:
                self.ml_tasks[key]='classification'
            elif val==float:
                self.ml_tasks[key]='regression'
        return self.ml_tasks


    def _prepare_data(self,embedding_name, property_name):
        property_type=self.property_types_map[property_name]
        property_df=self.df[[property_name,embedding_name]].dropna()
        ml_task=self.ml_tasks[property_name]
        class_names=None
        if ml_task=='classification':
            class_names=property_df[property_name].unique()

        property_df['X'] = property_df[embedding_name].apply(lambda x: np.array(x))
        X = np.array(property_df['X'].to_list())

        y=property_df[property_name]
        return X ,y,class_names,ml_task
    
    def _split_data(self,X,y,test_size=0.2,random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    
    def _train_model(self,X_train,y_train,X_test,y_test,ml_task,**kwargs):
        if ml_task=='classification':
            rf = RandomForestClassifier(**kwargs)
        elif ml_task=='regression':
            rf = RandomForestRegressor(**kwargs)

        #   Train the model
        rf.fit(X_train, y_train)

        y_pred, metrics = self._predict(rf,X_test,y_test, ml_task)
        
        return metrics
    
    def _predict(self,model,X_test,y_test,ml_task):
        metrics={}
        accuracy=None
        precision=None
        recall=None
        f1=None
        mse=None
        r2=None
        if ml_task=='classification':
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
        elif ml_task=='regression':
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

        metrics['accuracy']=accuracy
        metrics['precision']=precision
        metrics['recall']=recall
        metrics['f1']=f1
        metrics['mse']=mse
        metrics['r2']=r2
        return y_pred, metrics
    
    def _save_metrics(self):
        with open(os.path.join(self.metrics_path),'w') as f:
            json.dump(self.metrics_dict, f, indent=4)

    def train_models(self,test_size=0.2,random_state=42):
        for embedding_property in self.embedding_properties:
            for property in self.properties:
                print("-"*200)
                print(f"Training {property} on {embedding_property}")
                X,y,class_names,ml_task=self._prepare_data(embedding_property,property)
                X_train, X_test, y_train, y_test = self._split_data(X,y,test_size=test_size,random_state=random_state)

                metrics=self._train_model(X_train,y_train,X_test,y_test,ml_task)
                self.metrics_dict[property][embedding_property]=metrics
                print(metrics)
                self._save_metrics()



def main():
    # load_graphs_into_neo4j(from_scratch=False)

    # train_all_graph_node_embeddings(from_scratch=False)
    # train_experiment_node_embeddings(database_names=['main','spg-no-145','spg-no-196'],from_scratch=True)
    # embedding_properties=['fastrp-embedding-gc-chemenvElement','fastrp-embedding-ec-chemenvElement','fastrp-embedding-gec-chemenvElement',
    #                       'fastrp-embedding-gc-element-chemenv','fastrp-embedding-ec-element-chemenv','fastrp-embedding-gec-element-chemenv']
    # node_properties=['nsites','nelements','volume','density','density_atomic','composition_reduced','formula_pretty'
    #                  'e_electronic','e_ionic','e_total','energy_per_atom','energy_above_hull','formation_energy_per_atom',
    #                  'band_gap','vbm','cbm','efermi',
    #                  'crystal_system','space_group','point_group','hall_symbol'
    #                  'is_gap_direct','is_metal','is_magnetic','is_stable',
    #                  'ordering','total_magnetization','total_magnetization_normalized_vol','num_magnetic_sites','num_unique_magnetic_sites'
    #                  'g_ruess','g_voigt','g_vrh','k_reuss','k_voigt','k_vrh','homogeneous_poisson','universal_anisotropy']
    # embeddings_df, properties_df=get_embeddings_and_properties_df(database_name='nelements-2-2',node_type='Material',embedding_properties=embedding_properties, node_properties=node_properties)
    # print(embeddings_df.head())
    
    # print(properties_df['space_group'].unique())
    metrics_path='examples/scripts/base_metrics.json'
    benchmark=EmbeddingBenchmark(database_name='nelements-2-2',node_type='Material',metrics_path=metrics_path)
    benchmark.train_models(test_size=0.1)

    # print(node_types)
    # with Neo4jManager() as matgraphdb:
    #     manager=Neo4jGDSManager(matgraphdb)
    #     print(manager.get_graph_info(database_name=database_name,graph_name=graph_name))


if __name__=='__main__':
    main()