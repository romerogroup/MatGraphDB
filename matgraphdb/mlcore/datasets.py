from glob import glob
import os
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData,InMemoryDataset
import torch_geometric.transforms as T 

# from matgraphdb.mlcore.encoders import EdgeEncoders, NodeEncoders
from matgraphdb.mlcore.encoders import (CategoricalEncoder, ClassificationEncoder, IdentityEncoder,
                                        ListIdentityEncoder, BooleanEncoder, ElementsEncoder,
                                        CompositionEncoder,SpaceGroupOneHotEncoder)
from matgraphdb.graph.material_graph import MaterialGraph

from matgraphdb.utils import get_child_logger

logger=get_child_logger(__name__, console_out=False, log_level='debug')


# load node csv file and map index to continuous index
def load_node_csv(path, index_col, 
                  feature_encoders={}, 
                  target_encoders={}, 
                  node_filter={}, 
                  **kwargs):
    """Load a node csv file and map the index to a continuous index.

    Args:
        path (str): The path to the node csv file.
        index_col (str): The name of the index column.
        target_col (str, optional): The name of the target column.
        encoders (dict, optional): A dictionary mapping column names 
                                to encoders. Defaults to None.

    Returns:
        tuple: A tuple containing the encoded data and the mapping.
    """
    df = pd.read_csv(path, index_col=index_col, **kwargs)#.drop(axis=1, index=0)


    column_names=list(df.columns)


    if feature_encoders != {}:
        for name in feature_encoders.keys():
            if 'float[]' in name:
                values=[]
                for row in df[name]:
                    values.append([float(i) for i in row.split(';')])
                values = np.array(values)
                nan_mask = np.isnan(values)

                # Determine which rows contain at least one NaN
                rows_with_nan = nan_mask.any(axis=1)

                # Get indices of rows with at least one NaN
                row_indices = np.where(rows_with_nan)[0]

                if row_indices.size > 0:
                    # Drop rows that contain NaN values
                    df = df.drop(index=df.index[row_indices])

                
    
    if node_filter != {}:
        
        for key, value in node_filter.items():
            for name in column_names:
                if key in name:
                    column = name
                    break
            min_value, max_value = value
            df = df[(df[column] >= min_value) & (df[column] <= max_value)]

    target = None
    if target_encoders != None:
        ys=[]
        for col, encoder in target_encoders.items():
            df=df.dropna(subset=[col])
            ys.append(encoder(df[col]))

        # Checks if target_property exists in the dataframe
        if ys:
            target = torch.cat(ys, dim=-1)
    x = None
    if feature_encoders != {}:
        xs = [encoder(df[col]) for col, encoder in feature_encoders.items()]
        x = torch.cat(xs, dim=-1)
    
    names=df['name:string']
    # Create maping from original index to unqique index after removing posssible nan values
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    name_mapping = {i:name for i, name in enumerate(names)}
    return x, target, mapping, name_mapping

def load_edge_csv(path, src_index_col, dst_index_col, node_id_mappings, 
                  feature_encoders={}, 
                  target_encoders={},  
                  **kwargs):
    df = pd.read_csv(path, **kwargs)

    edge_type=os.path.basename(path).split('.')[0].lower()
    src_node_name,edge_type_name,dst_node_name=edge_type.split('-')
    src_mapping=node_id_mappings[src_node_name]
    dst_mapping=node_id_mappings[dst_node_name]



    edges_in_graph_mask=df[src_index_col].isin(src_mapping.keys()) & df[dst_index_col].isin(dst_mapping.keys())
    # filtered_df = df[edges_in_graph_mask]
    filtered_df = pd.DataFrame(df[edges_in_graph_mask].to_dict())

    edge_index=None
    edge_attr = None
    filtered_df['src_mapped'] = filtered_df[src_index_col].map(src_mapping)
    filtered_df['dst_mapped'] = filtered_df[dst_index_col].map(dst_mapping)

    edge_index = torch.tensor([filtered_df['src_mapped'].tolist(),
                               filtered_df['dst_mapped'].tolist()])
    

    target = None
    if target_encoders != None:
        ys=[]
        for col, encoder in target_encoders.items():
            df=df.dropna(subset=[col])
            ys.append(encoder(df[col]))

        # Checks if target_property exists in the dataframe
        if ys:
            target = torch.cat(ys, dim=-1)
    
    if feature_encoders != {}:
        edge_attrs = [encoder(filtered_df[col]) for col, encoder in feature_encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr, target

def load_node_parquet(path, columns=None):
    df = pd.read_parquet(path, columns=columns)

    print(df.head())
    columns=df.columns

    print(df['is_rare_earth'])
    cols_with_nan={ key:[] for key in columns}

    # for irow, row in df.iterrows():
    #     name=row['name']

    #     for col in columns:
    #         if row[col].isnull():
    #             cols_with_nan[col].append(name)
    cols_with_nan={}
    for col in columns:
        nan_rows=df[df[col].isnull()]

        cols_with_nan[col] = [row['name'] for irow, row in nan_rows.iterrows()]
    
    for col in cols_with_nan:
        print(col, cols_with_nan[col])

    # for col in columns:
        
    #     print(col , df[col].isnull().any())

        

        # xs = [encoder(df[col]) for col, encoder in feature_encoders.items()]
        # x = torch.cat(xs, dim=-1)


    return df

class PyTorchGeometricHeteroDataGenerator:
    def __init__(self):
        self.data = HeteroData()
        self.node_id_mappings={}

    def add_node_type(self, node_path, columns=None,
                feature_encoder_mapping={}, 
                target_encoder_mapping={}, 
                node_filter={}):
        


        node_name=os.path.basename(node_path).split('.')[0]
        df=load_node_parquet(node_path, columns=columns)
        # x, target, mapping, name_mapping = load_node_csv(node_path, 
        #                             index_col=0,
        #                             feature_encoders=feature_encoder_mapping,
        #                             target_encoders=target_encoder_mapping,
        #                             node_filter=node_filter)
        
        # self.data[node_name].node_id=torch.arange(len(mapping))
        # self.data[node_name].names=list(name_mapping.values())
        # if x is not None:
        #     self.data[node_name].x = x
        #     self.data[node_name].property_names=[key.split(':')[0] for key in list(feature_encoder_mapping.keys())]
        # else:
        #     self.data[node_name].num_nodes=len(mapping)

        # if target is not None:
        #     self.data[node_name].y_label_name=list(target_encoder_mapping.keys())[0].split(':')[0]
        #     out_channels=target.shape[1]
        #     self.data[node_name].out_channels = out_channels

        #     if out_channels==1:
        #         self.data[node_name].y=target
        #     else:
        #         self.data[node_name].y=torch.argmax(target, dim=1)

        # return mapping

    def add_edge(self, edge_path, node_id_mappings, 
                feature_encoder_mapping={},
                target_encoder_mapping={},
                undirected=True):
        
        graph_dir=os.path.dirname(os.path.dirname(edge_path))
        node_dir=os.path.join(graph_dir,'nodes')

        # Getting Node indicies and node type
        edge_name=os.path.basename(edge_path).split('.')[0].lower()
        src_name,edge_type,dst_name=edge_name.split('-')

        src_path=os.path.join(node_dir,f'{src_name}.csv')
        dst_path=os.path.join(node_dir,f'{dst_name}.csv')

        src_df=pd.read_csv(src_path,index_col=0)
        dst_df=pd.read_csv(dst_path,index_col=0)

        src_node_id_name = src_df.index.name.strip(')').split('(')[-1]
        dst_node_id_name = dst_df.index.name.strip(')').split('(')[-1]

        src_index_col=f":START_ID({src_node_id_name})"
        dst_index_col=f":END_ID({dst_node_id_name})"


        edge_index, edge_attr, target = load_edge_csv(edge_path, 
                                            src_index_col, 
                                            dst_index_col,
                                            node_id_mappings,
                                            feature_encoders=feature_encoder_mapping,
                                            target_encoders=target_encoder_mapping)

        self.data[src_name,edge_type,dst_name].edge_index=edge_index
        self.data[src_name,edge_type,dst_name].edge_attr=edge_attr

        self.data[src_name,edge_type,dst_name].property_names=[key.split(':')[0] for key in list(feature_encoder_mapping.keys())]

        if target is not None:
            self.data[src_name,edge_type,dst_name].y_label_name=list(target_encoder_mapping.keys())[0].split(':')[0]
            out_channels=target.shape[1]
            self.data[src_name,edge_type,dst_name].out_channels = out_channels

            if out_channels==1:
                self.data[src_name,edge_type,dst_name].y=target
            else:
                self.data[src_name,edge_type,dst_name].y=torch.argmax(target, dim=1)


        if undirected:
            row, col = edge_index
            rev_edge_index = torch.stack([col, row], dim=0)
            self.data[dst_name,f'rev_{edge_type}',src_name].edge_index=rev_edge_index
            self.data[dst_name,f'rev_{edge_type}',src_name].edge_attr=edge_attr

            if target is not None:
                self.data[dst_name,f'rev_{edge_type}',src_name].y_label_name=list(target_encoder_mapping.keys())[0].split(':')[0]
                out_channels=target.shape[1]
                self.data[dst_name,f'rev_{edge_type}',src_name].out_channels = out_channels


                if out_channels==1:
                    self.data[src_name,edge_type,dst_name].y=target
                else:
                    self.data[src_name,edge_type,dst_name].y=torch.argmax(target, dim=1)

    def add_node_types(self, 
                node_paths, 
                node_properties:dict={},
                node_filtering:dict={},
                target_property:str=None):
        
        node_encoders=NodeEncoders()
        for i,node_path in enumerate(node_paths):
            node_name=os.path.basename(node_path).split('.')[0]

            
            # Get the node filter for the node
            if node_name in node_filtering.keys():
                node_filter=node_filtering[node_name]
            else:
                node_filter={}

            # Get the properties for the node that are in node_properties
            if node_name in node_properties.keys():
                properties_to_keep=node_properties[node_name]['properties']
            else:
                properties_to_keep=[]
                
            # Get the encoder mapping for the node if use_node_properties is True
            feature_encoder_mapping={}
            if properties_to_keep!=[]:
                feature_encoder_mapping,_,_=node_encoders.get_encoder(node_path,**node_properties[node_name]['scale'])
                keys=list(feature_encoder_mapping.keys())
                for key in keys:
                    property_name=key.split(':')[0]
                    if property_name not in properties_to_keep:
                        feature_encoder_mapping.pop(key)

            # Get the target encoder mapping for the node if target_property is not None
            if target_property is not None:
                target_encoder_mapping,_,_=node_encoders.get_encoder(node_path)
                keys=list(target_encoder_mapping.keys())
                for key in keys:
                    property_name=key.split(':')[0]
                    if property_name != target_property and target_property:
                        target_encoder_mapping.pop(key)

                
            # Add the node to the graph
            mapping=self.add_node(
                                node_path, 
                                feature_encoder_mapping=feature_encoder_mapping,
                                target_encoder_mapping=target_encoder_mapping,
                                node_filter=node_filter,
                                )
            self.node_id_mappings.update({node_name:mapping})

        return None
    
    def add_relationships(self, relationship_paths, edge_properties, target_property:str=None, undirected=True):
        if self.node_id_mappings=={}:
            raise Exception("Node ID mappings must be created before adding relationships. Call add_nodes first")
        
        edge_encoders=EdgeEncoders()
        for i,relationship_path in enumerate(relationship_paths):

            feature_encoder_mapping={}
            if len(list(edge_properties.keys()))!=0:
                feature_encoder_mapping=edge_encoders.get_weight_edge_encoder(relationship_path,
                                                                            **edge_properties['weight']['scale'])
            

            # Get the target encoder mapping for the node if target_property is not None
            target_encoder_mapping={}
            if target_property is not None:
                target_encoder_mapping=edge_encoders.get_weight_edge_encoder(relationship_path)
                keys=list(target_encoder_mapping.keys())
                for key in keys:
                    property_name=key.split(':')[0]
                    if property_name != target_property and target_property:
                        target_encoder_mapping.pop(key)

            self.add_edge(relationship_path,
                        self.node_id_mappings,
                        feature_encoder_mapping=feature_encoder_mapping,
                        target_encoder_mapping=target_encoder_mapping,
                        undirected=undirected)
        return None



# class MaterialGraphDataset:
#     MAIN_GRAPH_DIR = GraphGenerator().main_graph_dir
#     MAIN_NODES_DIR = os.path.join(MAIN_GRAPH_DIR,'nodes')
#     MAIN_RELATIONSHIP_DIR = os.path.join(MAIN_GRAPH_DIR,'relationships')

#     def __init__(self,data):
#         self.data=data

#     @classmethod
#     def ec_element_chemenv(cls, sub_graph_path=None, 
#                                 node_properties:dict={},
#                                 node_filtering:dict={},
#                                 edge_properties:dict={},
#                                 node_target_property:str=None,
#                                 edge_target_property:str=None,
#                                 undirected=True):
        
#         if sub_graph_path is None:
#             node_dir=cls.MAIN_NODES_DIR
#             relationship_dir=cls.MAIN_RELATIONSHIP_DIR
#         else:
#             node_dir=os.path.join(sub_graph_path,'nodes')
#             relationship_dir=os.path.join(sub_graph_path,'relationships')

#         node_names=['element','chemenv','material']
#         relationship_names=['electric_connects','has','can_occur']
#         node_paths=[os.path.join(node_dir,f'{node_name}.csv') for node_name in node_names]

#         # Get the relationship paths of the nodes
#         relationship_paths = cls.get_relationship_paths(node_names, 
#                                                         relationship_names, 
#                                                         relationship_dir)
        
#         generator=cls.initialize_generator(node_paths,relationship_paths,
#                             node_properties=node_properties,
#                             edge_properties=edge_properties,
#                             node_filtering=node_filtering,
#                             node_target_property=node_target_property,
#                             edge_target_property=edge_target_property,
#                             undirected=undirected)
        
        

#         return cls(generator.data)

#     @classmethod
#     def gc_element_chemenv(cls, sub_graph_path=None, 
#                                 node_properties:dict={},
#                                 node_filtering:dict={},
#                                 edge_properties:dict={},
#                                 node_target_property:str=None,
#                                 edge_target_property:str=None,
#                                 undirected=True):
        
#         if sub_graph_path is None:
#             node_dir=cls.MAIN_NODES_DIR
#             relationship_dir=cls.MAIN_RELATIONSHIP_DIR
#         else:
#             node_dir=os.path.join(sub_graph_path,'nodes')
#             relationship_dir=os.path.join(sub_graph_path,'relationships')

#         node_names=['element','chemenv','material']
#         relationship_names=['geometric_connects','has','can_occur']
#         node_paths=[os.path.join(node_dir,f'{node_name}.csv') for node_name in node_names]

#         # Get the relationship paths of the nodes
#         relationship_paths = cls.get_relationship_paths(node_names, 
#                                                         relationship_names, 
#                                                         relationship_dir)

#         generator=cls.initialize_generator(node_paths,relationship_paths,
#                             node_properties=node_properties,
#                             edge_properties=edge_properties,
#                             node_filtering=node_filtering,
#                             node_target_property=node_target_property,
#                             edge_target_property=edge_target_property,
#                             undirected=undirected)

#         return cls(generator.data)
    
#     @classmethod
#     def gec_element_chemenv(cls, 
#                             sub_graph_path=None, 
#                             node_properties:dict={},
#                             node_filtering:dict={},
#                             edge_properties:dict={},
#                             node_target_property:str=None,
#                             edge_target_property:str=None,
#                             undirected=True):
        
#         if sub_graph_path is None:
#             node_dir=cls.MAIN_NODES_DIR
#             relationship_dir=cls.MAIN_RELATIONSHIP_DIR
#         else:
#             node_dir=os.path.join(sub_graph_path,'nodes')
#             relationship_dir=os.path.join(sub_graph_path,'relationships')

#         node_names=['element','chemenv','material']
#         relationship_names=['geometric_electric_connects','has','can_occur']
#         node_paths=[os.path.join(node_dir,f'{node_name}.csv') for node_name in node_names]
        
#         # Get the relationship paths of the nodes
#         relationship_paths = cls.get_relationship_paths(node_names, 
#                                                         relationship_names, 
#                                                         relationship_dir)

#         generator=cls.initialize_generator(node_paths,relationship_paths,
#                             node_properties=node_properties,
#                             edge_properties=edge_properties,
#                             node_filtering=node_filtering,
#                             node_target_property=node_target_property,
#                             edge_target_property=edge_target_property,
#                             undirected=undirected)

#         return cls(generator.data)

#     @staticmethod
#     def get_relationship_paths(node_names, relationship_names, relationship_dir):
#         """ Get the paths to the relationship files corredponding to the node used in the graph. """
#         relationship_files=glob(os.path.join(relationship_dir,'*.csv'))

#         relationship_paths=[]
#         for relationship_file in relationship_files:
#             edge_name=os.path.basename(relationship_file).split('.')[0].lower()
#             src_name,edge_type,dst_name=edge_name.split('-')
#             if src_name in node_names and dst_name in node_names and edge_type in relationship_names:
#                 relationship_paths.append(relationship_file)
#         return relationship_paths
    
#     @staticmethod
#     def initialize_generator(node_paths,relationship_paths,
#                             node_properties:dict,
#                             edge_properties:dict,
#                             node_filtering:dict,
#                             node_target_property:str,
#                             edge_target_property:str,
#                             undirected:bool=True):
        
#         generator=PyTorchGeometricHeteroDataGenerator()
#         generator.add_nodes(node_paths, 
#                             node_properties=node_properties, 
#                             node_filtering=node_filtering,
#                             target_property=node_target_property)
#         generator.add_relationships(relationship_paths, 
#                                     edge_properties, 
#                                     edge_target_property, 
#                                     undirected)
#         return generator


if __name__ == "__main__":
    

    import pandas as pd
    import os
    material_graph=MaterialGraph(skip_init=False)
    graph_dir = material_graph.graph_dir
    nodes_dir = material_graph.node_dir
    relationship_dir = material_graph.relationship_dir
 

    node_names=material_graph.list_nodes()
    relationship_names=material_graph.list_relationships()

    node_files=material_graph.get_node_filepaths()
    relationship_files=material_graph.get_relationship_filepaths()

    node_path=node_files[2]

    df=pd.read_parquet(node_path)

    df.to_csv(node_path.replace('.parquet','.csv'))

    # # Example of using pyGHeteroDataGenerator
    # generator=PyTorchGeometricHeteroDataGenerator()

    # node_properties={
    #     'ELEMENT':[
    #         'group',
    #         'row',
    #         'Z',
    #         # 'symbol',
    #         # 'long_name',
    #         'A',
    #         'atomic_radius_calculated',
    #         'van_der_waals_radius',
    #         'mendeleev_no',
    #         'electrical_resistivity',
    #         'velocity_of_sound',
    #         'reflectivity',
    #         'refractive_index',
    #         'poissons_ratio',
    #         'molar_volume',
    #         'thermal_conductivity',
    #         'boiling_point',
    #         'melting_point',
    #         'critical_temperature',
    #         'superconduction_temperature',
    #         'liquid_range',
    #         'bulk_modulus',
    #         'youngs_modulus',
    #         'rigidity_modulus',
    #         'vickers_hardness',
    #         'density_of_solid',
    #         'coefficient_of_linear_thermal_expansion',
    #         'block',
    #         'electron_affinity',
    #         'X',
    #         'atomic_mass',
    #         'atomic_mass_number',
    #         'atomic_radius',
    #         'average_anionic_radius',
    #         'average_cationic_radius',
    #         'average_ionic_radius',
    #         'ground_state_term_symbol',
    #         'is_actinoid',
    #         'is_alkali',
    #         'is_alkaline',
    #         'is_chalcogen',
    #         'is_halogen',
    #         'is_lanthanoid',
    #         'is_metal',
    #         'is_metalloid',
    #         'is_noble_gas',
    #         'is_post_transition_metal',
    #         'is_quadrupolar',
    #         'is_rare_earth',
    #         'is_rare_earth_metal',
    #         'is_transition_metal',
    #         'iupac_ordering',
    #         'max_oxidation_state',
    #         'min_oxidation_state',
    #         'valence',
    #         'name',
    #         # 'type',
    #     ]
    # }

    # generator.add_node_type(node_path=node_path, columns=node_properties['ELEMENT'])
    # encoder_mapping,_,_=node_encoders.get_element_encoder(element_node_path)
    # generator.add_node(element_node_path,encoder_mapping=encoder_mapping )
    # # generator.add_edge(material_crystal_system_path)
    # print(generator.data)

    # node_filtering={}


    ################################################################################################
    # Example of using MaterialGraphDataset
    
    # node_filtering={
    #     'material':{
    #         'k_vrh':(0,300),
    #         },
    #     }
    # node_properties={
    # 'element':
    #     {
    #     'properties' :[
    #             'atomic_number',
    #             'group',
    #             'row',
    #             'atomic_mass'
    #             ],
    #     'scale': {
    #             # 'robust_scale': True,
    #             # 'standardize': True,
    #             'normalize': True
    #         }
    #     },
    # 'material':
    #         {   
    #     'properties':[
    #         'nelements',
    #         'nsites',
    #         'crystal_system',
    #         'volume',
    #         'density',
    #         'density_atomic',
    #         ],
    #     'scale': {
    #             # 'robust_scale': True,
    #             'standardize': True,
    #             # 'normalize': True
    #         }
    #         }
    #     }

    # edge_properties={
    #     'weight':
    #         {
    #         'properties':[
    #             'weight'
    #             ],
    #         'scale': {
    #             # 'robust_scale': True,
    #             # 'standardize': True,
    #             # 'normalize': True
    #         }
    #     }
    #     }

    
    # target_property='k_vrh'
    # # Example of using MaterialGraphDataset
    # graph_dataset=MaterialGraphDataset.ec_element_chemenv(
    #                                                     node_properties=node_properties,
    #                                                     node_filtering=node_filtering,
    #                                                     edge_properties=edge_properties,
    #                                                     node_target_property=target_property,
    #                                                     edge_target_property=None,
    #                                                     )
    
    # data=graph_dataset.data

    # print(data)
    # print(data['element'].x[:10])    


    # print('material node target property')
    # print(f"Min: {data['material'].y.min()} | Max: {data['material'].y.max()} | Mean: {data['material'].y.mean()} | Std: {data['material'].y.std()} | Median: {data['material'].y.median()}")
    # print('-'*200)

    # print('element node')
    # print(data['element'].property_names)
    # for icol in range(data['element'].x.shape[1]):
    #     print(f"Column {icol} | Min: {data['element'].x[:,icol].min()} | Max: {data['element'].x[:,icol].max()} | Mean: {data['element'].x[:,icol].mean()} | Std: {data['element'].x[:,icol].std()} | Median: {data['element'].x[:,icol].median()}")
    # print('-'*200)

    # print('material node')
    # print(data['material'].property_names)
    # for icol in range(data['material'].x.shape[1]):
    #     print(f"Column {icol} | Min: {data['material'].x[:,icol].min()} | Max: {data['material'].x[:,icol].max()} | Mean: {data['material'].x[:,icol].mean()} | Std: {data['material'].x[:,icol].std()} | Median: {data['material'].x[:,icol].median()}")
    # print('-'*200)

    # print('element','electric_connects','element')
    # edge_attr=data['element','electric_connects','element'].edge_attr
    # print(data['element','electric_connects','element'].property_names)
    # for icol in range(edge_attr.shape[1]):
    #     print(f"Column {icol} | Min: {edge_attr[:,icol].min()} | Max: {edge_attr[:,icol].max()} | Mean: {edge_attr[:,icol].mean()} | Std: {edge_attr[:,icol].std()} | Median: {edge_attr[:,icol].median()}")
    # print('-'*200)



    # print(dir(graph_dataset.data))


    # # print(graph_dataset.data.edge_stores)
    # print(graph_dataset.data.edge_items())
    # # print(graph_dataset.data.node_items())
    # # print(graph_dataset.data.node_types)
    # print(graph_dataset.data.edge_types)
    