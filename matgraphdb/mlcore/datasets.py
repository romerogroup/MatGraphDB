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
def load_node_csv(path, index_col,  encoders={}, target_encoders={}, dropna=True, **kwargs):
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
    if dropna:
        df=df.dropna()
    
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    
    x = None
    if encoders != {}:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    target = None
    if target_encoders != None:
        ys = [encoder(df[col]) for col, encoder in target_encoders.items()]
        # Checks if target_property exists in the dataframe
        if ys:
            target = torch.cat(ys, dim=-1)

    return x, mapping, target

def load_edge_csv(path, src_index_col, dst_index_col, encoders={}, dropna=False,
                  src_mapping=None, dst_mapping=None,**kwargs):
    df = pd.read_csv(path, **kwargs)
    if dropna:
        df=df.dropna(axis=0)
    src = [ src_mapping[index] if src_mapping else index for index in df[src_index_col]]
    dst = [ dst_mapping[index] if dst_mapping else index for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders != {}:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr 

class PyTorchGeometricHeteroDataGenerator:
    def __init__(self,):
        self.data = HeteroData()

    def add_node(self, node_path, encoder_mapping={}, target_encoder_mapping={}, dropna=True, **kwargs):

        node_name=os.path.basename(node_path).split('.')[0]

        x, mapping, target = load_node_csv(node_path, 
                                    index_col="name:string",
                                    encoders=encoder_mapping,
                                    target_encoders=target_encoder_mapping,
                                    dropna=dropna, 
                                    **kwargs)
        
        self.data[node_name].node_id=torch.arange(len(mapping))
        if x is not None:
            self.data[node_name].x = x
            self.data[node_name].property_names=[key.split(':')[0] for key in list(encoder_mapping.keys())]
        else:
            self.data[node_name].num_nodes=len(mapping)

        if target is not None:
            self.data[node_name].y_label_name=list(target_encoder_mapping.keys())[0].split(':')[0]
            self.data[node_name].y=target

    def add_edge(self,edge_path, encoder_mapping={}, undirected=True, **kwargs):
        
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

        if encoder_mapping!={}:
            encoder_mapping=EdgeEncoders().get_weight_edge_encoder(edge_path)

        edge_index, edge_attr = load_edge_csv(edge_path, 
                                            src_index_col, 
                                            dst_index_col, 
                                            encoders=encoder_mapping, 
                                            **kwargs)

        self.data[src_name,edge_type,dst_name].edge_index=edge_index
        self.data[src_name,edge_type,dst_name].edge_attr=edge_attr

        if undirected:
            row, col = edge_index
            rev_edge_index = torch.stack([col, row], dim=0)
            self.data[dst_name,f'rev_{edge_type}',src_name].edge_index=rev_edge_index
            self.data[dst_name,f'rev_{edge_type}',src_name].edge_attr=edge_attr

    def to_undirected(self, in_place=True):
        data =  T.ToUndirected()(self.data)
        if in_place:
            self.data = data
        return data



class MaterialGraphDataset:
    MAIN_GRAPH_DIR = GraphGenerator().main_graph_dir
    MAIN_NODES_DIR = os.path.join(MAIN_GRAPH_DIR,'nodes')
    MAIN_RELATIONSHIP_DIR = os.path.join(MAIN_GRAPH_DIR,'relationships')

    def __init__(self,data):
        self.data=data

    @classmethod
    def ec_element_chemenv(cls, sub_graph_path=None, 
                                use_weights=True, 
                                use_node_properties=True, 
                                properties:Union[List,dict]=[],
                                target_property:str=None,
                                undirected=True):
        
        if sub_graph_path is None:
            node_dir=cls.MAIN_NODES_DIR
            relationship_dir=cls.MAIN_RELATIONSHIP_DIR
        else:
            node_dir=os.path.join(sub_graph_path,'nodes')
            relationship_dir=os.path.join(sub_graph_path,'relationships')

        node_names=['element','chemenv','material']
        relationship_names=['electric_connects','has','can_occur']
        dropnas=[False,False,False]
        node_paths=[os.path.join(node_dir,f'{node_name}.csv') for node_name in node_names]

        relationship_paths = cls.get_relationship_paths(node_names, 
                                                        relationship_names, 
                                                        relationship_dir)

        generator=PyTorchGeometricHeteroDataGenerator()

        cls.add_nodes(generator, node_paths,  use_node_properties, properties, dropnas, target_property)
        cls.add_relationships(generator, relationship_paths, use_weights, undirected)

        # if undirected:
        #     generator.to_undirected()

        return cls(generator.data)

    @classmethod
    def gc_element_chemenv(cls, sub_graph_path=None, 
                                use_weights=True, 
                                use_node_properties=True, 
                                properties:Union[List,dict]=[],
                                undirected=True):
        
        if sub_graph_path is None:
            node_dir=cls.MAIN_NODES_DIR
            relationship_dir=cls.MAIN_RELATIONSHIP_DIR
        else:
            node_dir=os.path.join(sub_graph_path,'nodes')
            relationship_dir=os.path.join(sub_graph_path,'relationships')

        node_names=['element','chemenv','material']
        relationship_names=['geometric_connects','has']
        node_paths=[os.path.join(node_dir,f'{node_name}.csv') for node_name in node_names]

        
        relationship_paths = cls.get_relationship_paths(node_names, relationship_names, relationship_dir)

        generator=PyTorchGeometricHeteroDataGenerator()
        cls.add_nodes(generator, node_paths,  use_node_properties, properties)
        cls.add_relationships(generator, relationship_paths, use_weights, undirected)

        # if undirected:
        #     generator.to_undirected()

        return cls(generator.data)

    @classmethod
    def gec_element_chemenv(cls, sub_graph_path=None, 
                                use_weights=True, 
                                use_node_properties=True,
                                properties:Union[List,dict]=[],
                                undirected=True):
        
        if sub_graph_path is None:
            node_dir=cls.MAIN_NODES_DIR
            relationship_dir=cls.MAIN_RELATIONSHIP_DIR
        else:
            node_dir=os.path.join(sub_graph_path,'nodes')
            relationship_dir=os.path.join(sub_graph_path,'relationships')

        node_names=['element','chemenv','material']
        relationship_names=['geometric_electric_connects','has']
        node_paths=[os.path.join(node_dir,f'{node_name}.csv') for node_name in node_names]

        relationship_paths = cls.get_relationship_paths(node_names, relationship_names, relationship_dir)

        generator=PyTorchGeometricHeteroDataGenerator()
        cls.add_nodes(generator, node_paths,  use_node_properties, properties)
        cls.add_relationships(generator, relationship_paths, use_weights, undirected)
    
        # if undirected:
        #     generator.to_undirected()

        return cls(generator.data)

    @staticmethod
    def get_relationship_paths(node_names, relationship_names, relationship_dir):
        relationship_files=glob(os.path.join(relationship_dir,'*.csv'))

        relationship_paths=[]
        for relationship_file in relationship_files:
            edge_name=os.path.basename(relationship_file).split('.')[0].lower()
            src_name,edge_type,dst_name=edge_name.split('-')
            if src_name in node_names and dst_name in node_names and edge_type in relationship_names:
                relationship_paths.append(relationship_file)
        return relationship_paths
    
    @staticmethod
    def add_nodes(generator, 
                node_paths, 
                use_node_properties, 
                properties:Union[List,dict], 
                dropnas:List[bool],
                target_property:str=None):
        node_encoders=NodeEncoders()
        for i,node_path in enumerate(node_paths):
            node_name=os.path.basename(node_path).split('.')[0]

            if isinstance(properties,dict):
                properties_to_keep=properties[node_name]
            else:
                properties_to_keep=properties

            encoder_mapping={}
            if use_node_properties:
                encoder_mapping,_,_=node_encoders.get_encoder(node_path)

                keys=list(encoder_mapping.keys())
                for key in keys:
                    property_name=key.split(':')[0]
                    if property_name not in properties_to_keep and properties_to_keep!=[]:
                        encoder_mapping.pop(key)

            if target_property is not None:
                target_encoder_mapping,_,_=node_encoders.get_encoder(node_path)
                keys=list(target_encoder_mapping.keys())
                for key in keys:
                    property_name=key.split(':')[0]
                    if property_name != target_property and target_property:
                        target_encoder_mapping.pop(key)

                
                
            generator.add_node(node_path, encoder_mapping=encoder_mapping, target_encoder_mapping=target_encoder_mapping, dropna=dropnas[i])
        return None
    
    @staticmethod
    def add_relationships(generator, relationship_paths, use_weights, undirected):
        edge_encoders=EdgeEncoders()
        for i,relationship_path in enumerate(relationship_paths):
            encoder_mapping={}
            if use_weights:
                encoder_mapping=edge_encoders.get_weight_edge_encoder(relationship_path)
            generator.add_edge(relationship_path,
                                encoder_mapping=encoder_mapping, 
                                undirected=undirected)
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

    # Example of using MaterialGraphDataset
    graph_dataset=MaterialGraphDataset.ec_element_chemenv(
                                                        use_weights=False,
                                                        use_node_properties=True,
                                                        undirected=True,
                                                        properties=['atomic_number','group','row','atomic_mass'],
                                                        target_property='formation_energy_per_atom'
                                                        )
                                                        
    print(graph_dataset.data)
    print(dir(graph_dataset.data))


    # # print(graph_dataset.data.edge_stores)
    # print(graph_dataset.data.edge_items())
    # # print(graph_dataset.data.node_items())
    # # print(graph_dataset.data.node_types)
    # print(graph_dataset.data.edge_types)
    