from glob import glob
import os
from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData,InMemoryDataset
import torch_geometric.transforms as T 

from matgraphdb.mlcore.encoders import EdgeEncoders, NodeEncoders
from matgraphdb import GraphGenerator

class NumpyDataset(Dataset):
    def __init__(self, X,y):
        self.X = X
        self.y = y
        self.n_samples=len(self.X)

    def __getitem__(self, index):
        features = self.X[index,:]
        label = self.y[index]
        return features, label
    
    def __len__(self):
        return self.n_samples

class PyTorchGeometricDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        data = self.data[index]
        return data

    def __len__(self):
        return len(self.data)


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

class PyTorchGeometricHeteroDataGenerator:
    def __init__(self):
        self.data = HeteroData()
        self.node_id_mappings={}

    def add_node(self, node_path, 
                feature_encoder_mapping={}, 
                target_encoder_mapping={}, 
                node_filter={}):

        node_name=os.path.basename(node_path).split('.')[0]




        x, target, mapping, name_mapping = load_node_csv(node_path, 
                                    index_col=0,
                                    feature_encoders=feature_encoder_mapping,
                                    target_encoders=target_encoder_mapping,
                                    node_filter=node_filter)
        
        self.data[node_name].node_id=torch.arange(len(mapping))
        self.data[node_name].names=list(name_mapping.values())
        if x is not None:
            self.data[node_name].x = x
            self.data[node_name].property_names=[key.split(':')[0] for key in list(feature_encoder_mapping.keys())]
        else:
            self.data[node_name].num_nodes=len(mapping)

        if target is not None:
            self.data[node_name].y_label_name=list(target_encoder_mapping.keys())[0].split(':')[0]
            out_channels=target.shape[1]
            self.data[node_name].out_channels = out_channels

            if out_channels==1:
                self.data[node_name].y=target
            else:
                self.data[node_name].y=torch.argmax(target, dim=1)

        return mapping

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

    def add_nodes(self, 
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



class MaterialGraphDataset:
    MAIN_GRAPH_DIR = GraphGenerator().main_graph_dir
    MAIN_NODES_DIR = os.path.join(MAIN_GRAPH_DIR,'nodes')
    MAIN_RELATIONSHIP_DIR = os.path.join(MAIN_GRAPH_DIR,'relationships')

    def __init__(self,data):
        self.data=data

    @classmethod
    def ec_element_chemenv(cls, sub_graph_path=None, 
                                node_properties:dict={},
                                node_filtering:dict={},
                                edge_properties:dict={},
                                node_target_property:str=None,
                                edge_target_property:str=None,
                                undirected=True):
        
        if sub_graph_path is None:
            node_dir=cls.MAIN_NODES_DIR
            relationship_dir=cls.MAIN_RELATIONSHIP_DIR
        else:
            node_dir=os.path.join(sub_graph_path,'nodes')
            relationship_dir=os.path.join(sub_graph_path,'relationships')

        node_names=['element','chemenv','material']
        relationship_names=['electric_connects','has','can_occur']
        node_paths=[os.path.join(node_dir,f'{node_name}.csv') for node_name in node_names]

        # Get the relationship paths of the nodes
        relationship_paths = cls.get_relationship_paths(node_names, 
                                                        relationship_names, 
                                                        relationship_dir)
        
        generator=cls.initialize_generator(node_paths,relationship_paths,
                            node_properties=node_properties,
                            edge_properties=edge_properties,
                            node_filtering=node_filtering,
                            node_target_property=node_target_property,
                            edge_target_property=edge_target_property,
                            undirected=undirected)
        
        

        return cls(generator.data)

    @classmethod
    def gc_element_chemenv(cls, sub_graph_path=None, 
                                node_properties:dict={},
                                node_filtering:dict={},
                                edge_properties:dict={},
                                node_target_property:str=None,
                                edge_target_property:str=None,
                                undirected=True):
        
        if sub_graph_path is None:
            node_dir=cls.MAIN_NODES_DIR
            relationship_dir=cls.MAIN_RELATIONSHIP_DIR
        else:
            node_dir=os.path.join(sub_graph_path,'nodes')
            relationship_dir=os.path.join(sub_graph_path,'relationships')

        node_names=['element','chemenv','material']
        relationship_names=['geometric_connects','has','can_occur']
        node_paths=[os.path.join(node_dir,f'{node_name}.csv') for node_name in node_names]

        # Get the relationship paths of the nodes
        relationship_paths = cls.get_relationship_paths(node_names, 
                                                        relationship_names, 
                                                        relationship_dir)

        generator=cls.initialize_generator(node_paths,relationship_paths,
                            node_properties=node_properties,
                            edge_properties=edge_properties,
                            node_filtering=node_filtering,
                            node_target_property=node_target_property,
                            edge_target_property=edge_target_property,
                            undirected=undirected)

        return cls(generator.data)
    
    @classmethod
    def gec_element_chemenv(cls, 
                            sub_graph_path=None, 
                            node_properties:dict={},
                            node_filtering:dict={},
                            edge_properties:dict={},
                            node_target_property:str=None,
                            edge_target_property:str=None,
                            undirected=True):
        
        if sub_graph_path is None:
            node_dir=cls.MAIN_NODES_DIR
            relationship_dir=cls.MAIN_RELATIONSHIP_DIR
        else:
            node_dir=os.path.join(sub_graph_path,'nodes')
            relationship_dir=os.path.join(sub_graph_path,'relationships')

        node_names=['element','chemenv','material']
        relationship_names=['geometric_electric_connects','has','can_occur']
        node_paths=[os.path.join(node_dir,f'{node_name}.csv') for node_name in node_names]
        
        # Get the relationship paths of the nodes
        relationship_paths = cls.get_relationship_paths(node_names, 
                                                        relationship_names, 
                                                        relationship_dir)

        generator=cls.initialize_generator(node_paths,relationship_paths,
                            node_properties=node_properties,
                            edge_properties=edge_properties,
                            node_filtering=node_filtering,
                            node_target_property=node_target_property,
                            edge_target_property=edge_target_property,
                            undirected=undirected)

        return cls(generator.data)

    @staticmethod
    def get_relationship_paths(node_names, relationship_names, relationship_dir):
        """ Get the paths to the relationship files corredponding to the node used in the graph. """
        relationship_files=glob(os.path.join(relationship_dir,'*.csv'))

        relationship_paths=[]
        for relationship_file in relationship_files:
            edge_name=os.path.basename(relationship_file).split('.')[0].lower()
            src_name,edge_type,dst_name=edge_name.split('-')
            if src_name in node_names and dst_name in node_names and edge_type in relationship_names:
                relationship_paths.append(relationship_file)
        return relationship_paths
    
    @staticmethod
    def initialize_generator(node_paths,relationship_paths,
                            node_properties:dict,
                            edge_properties:dict,
                            node_filtering:dict,
                            node_target_property:str,
                            edge_target_property:str,
                            undirected:bool=True):
        
        generator=PyTorchGeometricHeteroDataGenerator()
        generator.add_nodes(node_paths, 
                            node_properties=node_properties, 
                            node_filtering=node_filtering,
                            target_property=node_target_property)
        generator.add_relationships(relationship_paths, 
                                    edge_properties, 
                                    edge_target_property, 
                                    undirected)
        return generator


if __name__ == "__main__":
    

    import pandas as pd
    import os
    
    main_graph_dir = GraphGenerator().main_graph_dir
    main_nodes_dir = os.path.join(main_graph_dir,'nodes')
    main_relationship_dir = os.path.join(main_graph_dir,'relationships')
    print(main_graph_dir)

    element_node_path=os.path.join(main_nodes_dir,'element.csv')
    material_node_path=os.path.join(main_nodes_dir,'material.csv')
    material_crystal_system_path=os.path.join(main_relationship_dir,'MATERIAL-HAS-CRYSTAL_SYSTEM.csv')


    # Example of using pyGHeteroDataGenerator
    # node_encoders=NodeEncoders()
    # generator=PyTorchGeometricHeteroDataGenerator()
    # encoder_mapping,_,_=node_encoders.get_element_encoder(element_node_path)
    # generator.add_node(element_node_path,encoder_mapping=encoder_mapping )
    # # generator.add_edge(material_crystal_system_path)
    # print(generator.data)

    # node_filtering={}

    node_filtering={
        'material':{
            'k_vrh':(0,300),
            },
        }
    node_properties={
    'element':
        {
        'properties' :[
                'atomic_number',
                'group',
                'row',
                'atomic_mass'
                ],
        'scale': {
                # 'robust_scale': True,
                # 'standardize': True,
                'normalize': True
            }
        },
    'material':
            {   
        'properties':[
            'nelements',
            'nsites',
            'crystal_system',
            'volume',
            'density',
            'density_atomic',
            ],
        'scale': {
                # 'robust_scale': True,
                'standardize': True,
                # 'normalize': True
            }
            }
        }

    edge_properties={
        'weight':
            {
            'properties':[
                'weight'
                ],
            'scale': {
                # 'robust_scale': True,
                # 'standardize': True,
                # 'normalize': True
            }
        }
        }

    
    target_property='k_vrh'
    # Example of using MaterialGraphDataset
    graph_dataset=MaterialGraphDataset.ec_element_chemenv(
                                                        node_properties=node_properties,
                                                        node_filtering=node_filtering,
                                                        edge_properties=edge_properties,
                                                        node_target_property=target_property,
                                                        edge_target_property=None,
                                                        )
    
    data=graph_dataset.data

    # print(data)
    # print(data['element'].x[:10])    


    print('material node target property')
    print(f"Min: {data['material'].y.min()} | Max: {data['material'].y.max()} | Mean: {data['material'].y.mean()} | Std: {data['material'].y.std()} | Median: {data['material'].y.median()}")
    print('-'*200)

    print('element node')
    print(data['element'].property_names)
    for icol in range(data['element'].x.shape[1]):
        print(f"Column {icol} | Min: {data['element'].x[:,icol].min()} | Max: {data['element'].x[:,icol].max()} | Mean: {data['element'].x[:,icol].mean()} | Std: {data['element'].x[:,icol].std()} | Median: {data['element'].x[:,icol].median()}")
    print('-'*200)

    print('material node')
    print(data['material'].property_names)
    for icol in range(data['material'].x.shape[1]):
        print(f"Column {icol} | Min: {data['material'].x[:,icol].min()} | Max: {data['material'].x[:,icol].max()} | Mean: {data['material'].x[:,icol].mean()} | Std: {data['material'].x[:,icol].std()} | Median: {data['material'].x[:,icol].median()}")
    print('-'*200)

    print('element','electric_connects','element')
    edge_attr=data['element','electric_connects','element'].edge_attr
    print(data['element','electric_connects','element'].property_names)
    for icol in range(edge_attr.shape[1]):
        print(f"Column {icol} | Min: {edge_attr[:,icol].min()} | Max: {edge_attr[:,icol].max()} | Mean: {edge_attr[:,icol].mean()} | Std: {edge_attr[:,icol].std()} | Median: {edge_attr[:,icol].median()}")
    print('-'*200)
    # print(dir(graph_dataset.data))


    # # print(graph_dataset.data.edge_stores)
    # print(graph_dataset.data.edge_items())
    # # print(graph_dataset.data.node_items())
    # # print(graph_dataset.data.node_types)
    # print(graph_dataset.data.edge_types)
    