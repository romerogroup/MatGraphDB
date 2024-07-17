import os 

import torch

import numpy as np
import pandas as pd

from matgraphdb.mlcore.transforms import min_max_normalize, standardize_tensor, robust_scale

class CategoricalEncoder:
    def __init__(self, sep='|'):
        self.sep = sep

    def __call__(self, df):
        genres = set(g for col in df.values for g in col.split(self.sep))
        mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x

class ClassificationEncoder:
    """Converts a column of of unique itentities into a torch tensor. One hot encoding"""
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        # Find unique values in the column
        unique_values = df.unique()
        # Create a dictionary mapping unique values to integers
        value_to_index = {value: i for i, value in enumerate(unique_values)}
        tensor=torch.zeros(len(df),len(unique_values))

        for irow,elements in enumerate(df):
            tensor[irow,value_to_index[elements]]=1
        return tensor
    
class BooleanEncoder:
    """Converts a column of boolean values into a torch tensor."""
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        # Convert boolean values to integers (True to 1, False to 0)
        boolean_integers = df.astype(int)
        # Create a Torch tensor from the numpy array, ensure it has the correct dtype
        return torch.from_numpy(boolean_integers.values).view(-1, 1).type(self.dtype)
    
class IdentityEncoder:
    """Converts a column of numbers into torch tensor."""
    def __init__(self, dtype=torch.float32, standardize=False, normalize=False, robust_scale=False, normalization_range=(0,1)):
        self.dtype = dtype
        self.standardize = standardize
        self.normalize = normalize
        self.robust_scale = robust_scale
        self.normalization_range = normalization_range
        if self.standardize and self.normalize and self.robust_scale:
            raise Exception("Cannot standardize, normalize, and robust_scale at the same time")

    def __call__(self, df):
        tensor=torch.from_numpy(df.values).view(-1, 1).to(self.dtype)
        if self.standardize:
            tensor, mean, std = standardize_tensor(tensor)
            return tensor.to(self.dtype)
        elif self.robust_scale:
            tensor, median, iqr = robust_scale(tensor,q_min=0.25,q_max=0.75)
            return tensor.to(self.dtype)
        elif self.normalize:
            tensor, min, max = min_max_normalize(tensor,normalization_range=self.normalization_range)
            return tensor.to(self.dtype)
        else:
            return tensor
    
class ListIdentityEncoder:
    """Converts a column of list of numbers into torch tensor."""
    def __init__(self, dtype=None, standardize=False, normalize=False, robust_scale=False, normalization_range=(0,1)):
        self.dtype = dtype
        self.standardize = standardize
        self.normalize = normalize
        self.robust_scale = robust_scale
        self.normalization_range = normalization_range
        if self.standardize and self.normalize and self.robust_scale:
            raise Exception("Cannot standardize, normalize, and robust_scale at the same time")

    def __call__(self, df):
        values=[]
        for irow,row in enumerate(df):
            embedding=[float(i) for i in row.split(';')]
            values.append(embedding)
        values=np.array(values)

        tensor=torch.from_numpy(values).to(self.dtype)
        if self.standardize:
            tensor, mean, std = standardize_tensor(tensor)
            return tensor.to(self.dtype)
        elif self.robust_scale:
            tensor, median, iqr = robust_scale(tensor,q_min=0.25,q_max=0.75)
            return tensor.to(self.dtype)
        elif self.normalize:
            tensor, min, max = min_max_normalize(tensor,normalization_range=self.normalization_range)
            return tensor.to(self.dtype)
        else:
            return tensor

class ElementsEncoder:
    """Converts a column of list of numbers into torch tensor."""
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        from matgraphdb.utils.periodic_table import atomic_symbols
        tensor=torch.zeros(len(df),118)
        element_to_z={element:i-1 for i,element in enumerate(atomic_symbols)}
        for irow,elements in enumerate(df):
            elemnt_indices=[element_to_z[e] for e in elements.split(';')]
            tensor[irow,elemnt_indices]+=1
        return tensor

class CompositionEncoder:
    """Converts a column of list of numbers into torch tensor."""
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        from matgraphdb.utils.periodic_table import atomic_symbols
        import ast
        tensor=torch.zeros(len(df),118)
        element_to_z={element:i-1 for i,element in enumerate(atomic_symbols)}
        for irow,comp_string in enumerate(df):
            comp_mapping=ast.literal_eval(comp_string)
            for element,comp_val in comp_mapping.items():
                element_index=element_to_z[element]
                tensor[irow,element_index]+=comp_val
        # Normalize tensor by row
        tensor=tensor/tensor.sum(axis=1, keepdims=True)
        return tensor

class SpaceGroupOneHotEncoder:
    """Converts a column of list of numbers into torch tensor."""
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        tensor=torch.zeros(len(df),230)
        for irow,space_group in enumerate(df):
            tensor[irow,space_group-1]+=1
        return tensor

class NodeEncoders:

    ELEMENT_NODE='element'
    CHEMENV_NODE='chemenv'
    CRYSTAL_SYSTEM_NODE='crystal_system'
    MAGNETIC_STATE_NODE='magnetic_state'
    SPACE_GROUP_NODE='space_group'
    OXIDATION_STATE_NODE='oxidation_state'
    MATERIAL_NODE='material'

    def get_encoder(self,node_path, **kwargs):
        node_name=os.path.basename(node_path).split('.')[0]

        if node_name==self.ELEMENT_NODE:
            encoder_info =self.get_element_encoder(node_path, **kwargs)
            encoder_mapping, name_id_map, id_name_map = encoder_info

        if node_name==self.CHEMENV_NODE:
            encoder_info =self.get_chemenv_encoder(node_path, **kwargs)
            encoder_mapping, name_id_map, id_name_map = encoder_info

        if node_name==self.CRYSTAL_SYSTEM_NODE:
            encoder_info =self.get_crystal_system_encoder(node_path, **kwargs)
            encoder_mapping, name_id_map, id_name_map = encoder_info

        if node_name==self.MAGNETIC_STATE_NODE:
            encoder_info =self.get_magnetic_states_encoder(node_path, **kwargs)
            encoder_mapping, name_id_map, id_name_map = encoder_info

        if node_name==self.SPACE_GROUP_NODE:
            encoder_info =self.get_space_group_encoder(node_path, **kwargs)
            encoder_mapping, name_id_map, id_name_map = encoder_info

        if node_name==self.OXIDATION_STATE_NODE:
            encoder_info =self.get_oxidation_states_encoder(node_path, **kwargs)
            encoder_mapping, name_id_map, id_name_map = encoder_info

        if node_name==self.MATERIAL_NODE:
            encoder_info =self.get_material_encoder(node_path, **kwargs)
            encoder_mapping, name_id_map, id_name_map = encoder_info

        return encoder_mapping, name_id_map, id_name_map
   
    def get_encoder_mapping(self,node_path, skip_columns=[], column_encoders={}):
        df=pd.read_csv(node_path,index_col=0)

        name_id_map=df['name:string'].to_dict()
        id_name_map={i:name for i,name in enumerate(name_id_map)}
        encoder_mapping={}
        for col in df.columns:
            col_name=col.split(':')[0]
            col_type=col.split(':')[1]

            is_array_type=False
            if '[]' in col_type:
                col_type=col_type.split('[]')[0]
                is_array_type=True

            # Skip columns
            if col_name in skip_columns:
                continue
            
            # Special encoders for specific columns
            if col in column_encoders.keys():
                # print(f"Using column encoder for {col_name}")
                # print(column_encoders.keys())
                encoder_mapping[col]=column_encoders[col]
                continue

            # Defualt encoders based on column type that are not array types
            if col_type=='float' and not is_array_type:
                encoder_mapping[col]=IdentityEncoder(dtype=torch.float32)
            elif col_type=='int':
                encoder_mapping[col]=IdentityEncoder(dtype=torch.float32)
            elif col_type=='boolean':
                encoder_mapping[col]=BooleanEncoder(dtype=torch.int64)
            elif col_type=='string':
                encoder_mapping[col]=ClassificationEncoder(dtype=torch.int64)
            
            # Default encoders based on column type that are array types
            if col_type=='float' and is_array_type:
                encoder_mapping[col]=ListIdentityEncoder(dtype=torch.float32)
            elif col_type=='int' and is_array_type:
                encoder_mapping[col]=ListIdentityEncoder(dtype=torch.int64)

        return encoder_mapping, name_id_map, id_name_map
    
    def get_element_encoder(self,node_path, column_encoders={}, **kwargs):
        if column_encoders=={}:
            # column_encoders={
            #     'atomic_number:float':IdentityEncoder(normalize=do_scaling),
            #     'X:float':IdentityEncoder(normalize=do_scaling),
            #     'atomic_radius:float':IdentityEncoder(normalize=do_scaling),
            #     'group:int':IdentityEncoder(normalize=do_scaling),
            #     'row:int':IdentityEncoder(normalize=do_scaling),
            #     'atomic_mass:float':IdentityEncoder(normalize=do_scaling),
            #     }
            column_encoders={
                'atomic_number:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'X:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'atomic_radius:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'group:int':IdentityEncoder(dtype=torch.float32,**kwargs),
                'row:int':IdentityEncoder(dtype=torch.float32,**kwargs),
                'atomic_mass:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                }
        encoder_mapping,name_id_map, id_name_map=self.get_encoder_mapping(node_path=node_path,
                                                skip_columns=['name','type','element_name'],
                                                column_encoders=column_encoders)
        return encoder_mapping, name_id_map, id_name_map
    
    def get_chemenv_encoder(self,node_path, column_encoders={}, **kwargs):
        encoder_mapping, name_id_map, id_name_map=self.get_encoder_mapping(node_path=node_path,
                                                skip_columns=['name','type'])

        return encoder_mapping, name_id_map, id_name_map
    
    def get_chemenv_element_encoder(self,node_path, column_encoders={}, **kwargs):
        encoder_mapping, name_id_map, id_name_map=self.get_encoder_mapping(node_path=node_path,
                                                skip_columns=['name','type','chemenv_name'])
        return encoder_mapping, name_id_map, id_name_map
    
    def get_crystal_system_encoder(self,node_path, column_encoders={}, **kwargs):
        encoder_mapping, name_id_map, id_name_map=self.get_encoder_mapping(node_path=node_path,
                                                skip_columns=['name','type'])

        return encoder_mapping, name_id_map, id_name_map
    
    def get_crystal_system_encoder(self,node_path, column_encoders={}, **kwargs):
        encoder_mapping, name_id_map, id_name_map=self.get_encoder_mapping(node_path=node_path,
                                                skip_columns=['name','type'])

        return encoder_mapping, name_id_map, id_name_map
    
    def get_magnetic_states_encoder(self,node_path, column_encoders={}, **kwargs):
        encoder_mapping, name_id_map, id_name_map=self.get_encoder_mapping(node_path=node_path,
                                                skip_columns=['name','type'])
        return encoder_mapping, name_id_map, id_name_map
    
    def get_space_group_encoder(self,node_path, column_encoders={}, **kwargs):
        encoder_mapping, name_id_map, id_name_map=self.get_encoder_mapping(node_path=node_path,
                                                skip_columns=['name','type'])

        return encoder_mapping, name_id_map, id_name_map
    
    def get_oxidation_states_encoder(self,node_path, column_encoders={}, **kwargs):
        encoder_mapping, name_id_map, id_name_map=self.get_encoder_mapping(node_path=node_path,
                                                skip_columns=['name','type'])
        return encoder_mapping, name_id_map, id_name_map
    
    def get_material_encoder(self,node_path,skip_columns=[],column_encoders={}, **kwargs):
        if column_encoders=={}:
            # column_encoders={
            #     'composition:string':CompositionEncoder(),
            #     'elements:string[]':ElementsEncoder(),
            #     'crystal_system:string':ClassificationEncoder(),
            #     'space_group:int':SpaceGroupOneHotEncoder(),
            #     'point_group:string':ClassificationEncoder(),
            #     'hall_symbol:string':ClassificationEncoder(),
            #     'is_gap_direct:boolean':ClassificationEncoder(),
            #     'is_metal:boolean':ClassificationEncoder(),
            #     'is_magnetic:boolean':ClassificationEncoder(),
            #     'ordering:string':ClassificationEncoder(),
            #     'is_stable:boolean':ClassificationEncoder(),
                
            #     # 'nelements:int':ClassificationEncoder(),
            #     'nelements:int':IdentityEncoder(dtype=torch.float32,normalize=do_scaling),
            #     'nsites:int':IdentityEncoder(dtype=torch.float32,normalize=do_scaling),
            #     'volume:float':IdentityEncoder(dtype=torch.float32,robust_scale=do_scaling),
            #     'density:float':IdentityEncoder(dtype=torch.float32,normalize=do_scaling),
            #     'density_atomic:float':IdentityEncoder(dtype=torch.float32,normalize=do_scaling),
            #     'energy_per_atom:float':IdentityEncoder(dtype=torch.float32,normalize=do_scaling),
            #     'formation_energy_per_atom:float':IdentityEncoder(dtype=torch.float32,normalize=do_scaling),
            #     'energy_above_hull:float':IdentityEncoder(dtype=torch.float32,normalize=do_scaling),
            #     'band_gap:float':IdentityEncoder(dtype=torch.float32,normalize=do_scaling),

            #     'cbm:float':IdentityEncoder(dtype=torch.float32,robust_scale=do_scaling),
            #     'vbm:float':IdentityEncoder(dtype=torch.float32,robust_scale=do_scaling),

            #     'efermi:float':IdentityEncoder(dtype=torch.float32,normalize=do_scaling),
            #     'total_magnetization:float':IdentityEncoder(dtype=torch.float32,normalize=do_scaling),
            #     'total_magnetization_normalized_vol:float':IdentityEncoder(dtype=torch.float32,normalize=do_scaling),
            #     'num_magnetic_sites:int':IdentityEncoder(dtype=torch.float32,normalize=do_scaling),
            #     'num_unique_magnetic_sites:int':IdentityEncoder(dtype=torch.float32,normalize=do_scaling),

            #     'k_voigt:float':IdentityEncoder(dtype=torch.float32,robust_scale=do_scaling),
            #     'k_reuss:float':IdentityEncoder(dtype=torch.float32,robust_scale=do_scaling),
            #     'k_vrh:float':IdentityEncoder(dtype=torch.float32,robust_scale=do_scaling),

            #     'g_voigt:float':IdentityEncoder(dtype=torch.float32,robust_scale=do_scaling),
            #     'g_reuss:float':IdentityEncoder(dtype=torch.float32,robust_scale=do_scaling),
            #     'g_vrh:float':IdentityEncoder(dtype=torch.float32,robust_scale=do_scaling),

            #     'universal_anisotropy:float':IdentityEncoder(dtype=torch.float32,robust_scale=do_scaling),
            #     'homogeneous_poisson:float':IdentityEncoder(dtype=torch.float32,robust_scale=do_scaling),
            #     'e_total:float':IdentityEncoder(dtype=torch.float32,robust_scale=do_scaling),
            #     'e_ionic:float':IdentityEncoder(dtype=torch.float32,normalize=do_scaling),
            #     'e_electronic:float':IdentityEncoder(dtype=torch.float32,normalize=do_scaling),
            #     }
            
            column_encoders={
                'composition:string':CompositionEncoder(),
                'elements:string[]':ElementsEncoder(),
                'crystal_system:string':ClassificationEncoder(),
                'space_group:int':SpaceGroupOneHotEncoder(),
                'point_group:string':ClassificationEncoder(),
                'hall_symbol:string':ClassificationEncoder(),
                'is_gap_direct:boolean':ClassificationEncoder(),
                'is_metal:boolean':ClassificationEncoder(),
                'is_magnetic:boolean':ClassificationEncoder(),
                'ordering:string':ClassificationEncoder(),
                'is_stable:boolean':ClassificationEncoder(),
                
                # 'nelements:int':ClassificationEncoder(),
                'nelements:int':IdentityEncoder(dtype=torch.float32,**kwargs),
                'nsites:int':IdentityEncoder(dtype=torch.float32,**kwargs),
                'volume:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'density:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'density_atomic:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'energy_per_atom:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'formation_energy_per_atom:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'energy_above_hull:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'band_gap:float':IdentityEncoder(dtype=torch.float32,**kwargs),

                'cbm:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'vbm:float':IdentityEncoder(dtype=torch.float32,**kwargs),

                'efermi:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'total_magnetization:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'total_magnetization_normalized_vol:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'num_magnetic_sites:int':IdentityEncoder(dtype=torch.float32,**kwargs),
                'num_unique_magnetic_sites:int':IdentityEncoder(dtype=torch.float32,**kwargs),

                'k_voigt:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'k_reuss:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'k_vrh:float':IdentityEncoder(dtype=torch.float32,**kwargs),

                'g_voigt:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'g_reuss:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'g_vrh:float':IdentityEncoder(dtype=torch.float32,**kwargs),

                'universal_anisotropy:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'homogeneous_poisson:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'e_total:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'e_ionic:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                'e_electronic:float':IdentityEncoder(dtype=torch.float32,**kwargs),

                'sine_coulomb_matrix:float[]':ListIdentityEncoder(dtype=torch.float32,**kwargs),
                'element_fraction:float[]':ListIdentityEncoder(dtype=torch.float32,**kwargs),
                'element_property:float[]':ListIdentityEncoder(dtype=torch.float32,**kwargs),
                'xrd_pattern:float[]':ListIdentityEncoder(dtype=torch.float32,**kwargs),
                }
        encoder_mapping, name_id_map, id_name_map=self.get_encoder_mapping(node_path=node_path,
                                                skip_columns=skip_columns,
                                                column_encoders=column_encoders)
        return encoder_mapping, name_id_map, id_name_map
    
    
    
class EdgeEncoders:

    def get_encoder_mapping(self,edge_path, skip_columns=[], column_encoders={}):
        df=pd.read_csv(edge_path)

        encoder_mapping={}
        for col in df.columns:
            # Skip id columns

            if ":START_ID" in col:
                continue
            if ":END_ID" in col:
                continue
            if ":TYPE" in col:
                continue

            col_name=col.split(':')[0]
            col_type=col.split(':')[1]

            is_array_type=False
            if '[]' in col_type:
                col_type=col_type.split('[]')[0]
                is_array_type=True

            
            # Skip columns
            if col_name in skip_columns:
                continue
            
            # Special encoders for specific columns
            if col in column_encoders.keys():
                encoder_mapping[col]=column_encoders[col]
                continue


            # Defualt encoders based on column type that are not array types
            if col_type=='float' and not is_array_type:
                encoder_mapping[col]=IdentityEncoder(dtype=torch.float32)
            elif col_type=='int':
                encoder_mapping[col]=IdentityEncoder(dtype=torch.float32)
            elif col_type=='boolean':
                encoder_mapping[col]=BooleanEncoder(dtype=torch.int64)
            elif col_type=='string':
                encoder_mapping[col]=ClassificationEncoder(dtype=torch.int64)
            
            # Default encoders based on column type that are array types
            if col_type=='float' and is_array_type:
                encoder_mapping[col]=ListIdentityEncoder(dtype=torch.float32)
            elif col_type=='int' and is_array_type:
                encoder_mapping[col]=ListIdentityEncoder(dtype=torch.int64)

        return encoder_mapping
    
    def get_weight_edge_encoder(self,edge_path,column_encoders={}, **kwargs):
        if column_encoders=={}:
            column_encoders={
                'weight:float':IdentityEncoder(dtype=torch.float32,**kwargs),
                }
        encoder_mapping=self.get_encoder_mapping(edge_path=edge_path,column_encoders=column_encoders)
        return encoder_mapping



if __name__ == "__main__":
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    from matgraphdb import GraphGenerator
    from matgraphdb.mlcore.transforms import min_max_normalize, standardize_tensor
    main_graph_dir = GraphGenerator().main_graph_dir
    main_nodes_dir = os.path.join(main_graph_dir,'nodes') 
    print(main_graph_dir)

    element_node_path=os.path.join(main_nodes_dir,'element.csv')
    chemenv_node_path=os.path.join(main_nodes_dir,'chemenv.csv')
    material_node_path=os.path.join(main_nodes_dir,'material.csv')

    element_df=pd.read_csv(element_node_path,index_col=0)
    chemenv_df=pd.read_csv(chemenv_node_path,index_col=0)
    material_df=pd.read_csv(material_node_path,index_col=0)


    # Example usage of encoders
    # identity_encoder=IdentityEncoder()
    # atomic_numbers=identity_encoder(material_df['k_vrh:float'])
    # print(atomic_numbers.shape)


    # Example usage of encoders
    identity_encoder=ListIdentityEncoder(dtype=torch.float32,normalize=True)
    values=identity_encoder(material_df['element_property:float[]'])
    print(values[:10])
    # print(atomic_numbers.shape)

    # # Example usage of encoders
    # identity_encoder=IdentityEncoder(normalization_func=min_max_normalize)
    # atomic_numbers=identity_encoder(element_df['atomic_number:float'])
    # x=identity_encoder(element_df['X:float'])

    # Example of special encoder
    # list_encoder=ElementsEncoder()
    # tensor=list_encoder(material_df['elements:string[]'])
    # print(tensor.shape)
    # print(tensor[:10])

    # Example of special encoder
    # comp_encoder=CompositionEncoder()
    # tensor=comp_encoder(material_df['composition:string'])
    # print(tensor.shape)
    # print(tensor[:10])
    # print(tensor[0])

    # Example of Space Group encoder
    # space_group_encoder=SpaceGroupOneHotEncoder()
    # tensor=space_group_encoder(material_df['space_group:int'])
    # print(tensor.shape)
    # print(tensor[:10])

    # Example of using NodeEncoders
    # node_encoders=NodeEncoders()
    # encoders,_,_=node_encoders.get_element_encoder(element_node_path)
    # x = None
    # if encoders is not None:
    #     xs = [encoder(element_df[col]) for col, encoder in encoders.items()]
    #     x = torch.cat(xs, dim=-1)


    # Example of using NodeEncoders
    # node_encoders=NodeEncoders()
    # encoders,_,_=node_encoders.get_chemenv_encoder(chemenv_node_path)
    # x = None
    # if encoders != {}:
    #     print(encoders)
    #     xs = [encoder(chemenv_df[col]) for col, encoder in encoders.items()]
    #     x = torch.cat(xs, dim=-1)
    #     print(x.shape)

    
    # main_graph_dir = GraphGenerator.main_graph_dir
    # df = pd.read_csv()