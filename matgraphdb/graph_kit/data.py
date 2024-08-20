import io
import os

import pandas as pd
import pyarrow.parquet as pq
import torch
from torch_geometric.data import HeteroData

from matgraphdb.graph_kit.pyg.encoders import *
from matgraphdb.graph_kit.graphs import GraphManager
from matgraphdb.utils import get_child_logger

logger=get_child_logger(__name__, console_out=False, log_level='debug')

def get_parquet_field_metadata(path, columns=None):
    parquet_file = pq.ParquetFile(path)
    field_metadata={}
    for field in parquet_file.metadata.schema.to_arrow_schema():
        name=field.name

        if columns and name not in columns:
            continue

        field_metadata[name]={}
        for key,value in field.metadata.items():
            field_metadata[name][key.decode('utf-8')]=value.decode('utf-8')
    return field_metadata

def load_node_parquet(path, feature_columns=[], target_columns=[], custom_encoders={}, filter={}):
    if target_columns is None:
        target_columns=[]
    if feature_columns is None:
        feature_columns=[]

    

    all_columns=feature_columns+target_columns

    if 'name' not in all_columns:
        all_columns.append('name')

    # Getting field metadata for all columns
    field_metadata=get_parquet_field_metadata(path,columns=all_columns)

    # Reading all columns into single dataframe
    df = pd.read_parquet(path, columns=all_columns)
    names=df['name']
    df = df.drop(columns=['name'])
    all_columns.remove('name')
    column_names=list(df.columns)

    logger.info(f"Dataframe shape: {df.shape}")
    logger.info(f"Column names: {column_names}")

    # Ensure all columns have no NaN values, otherwise drop them
    df.dropna(subset=df.columns, inplace=True)

    logger.info(f"Dataframe shape after removing NaN values: {df.shape}")

    # Apply data filter to the nodes
    for key, (min_value, max_value) in filter.items():
        df = df[(df[key] >= min_value) & (df[key] <= max_value)]

    logger.info(f"Dataframe shape after applying node filter: {df.shape}")

    # Applying encoders to the nodes
    xs, ys = [], []
    feature_names, target_names = [], []
    for column_name in column_names:
        tmp_names=[]
        # Applying custom encoder if provided, 
        # otherwise use default encoder inside parquet feild metadata
        if column_name in custom_encoders:
            if isinstance(custom_encoders[column_name],str):
                encoder=eval(custom_encoders[column_name])
            else:
                encoder=custom_encoders[column_name]
        else:
            encoder=eval(field_metadata[column_name]['encoder'])

        encoded_values=encoder(df[column_name])

        # Getting feature names from encoder. Some encoders return multiple feature 
        # columns due to the nature of the encoder
        encoder_feature_names=None
        if hasattr(encoder,'column_names'):
            encoder_feature_names=encoder.column_names

        # Generating feature names
        if encoder_feature_names:
            for feature_name in encoder_feature_names:
                tmp_names.append(f"{column_name}_{feature_name}")
        elif encoded_values.shape[1]>1:
            for i in range(encoded_values.shape[1]):
                tmp_names.append(f"{column_name}_{i}")
        else:
            tmp_names.append(column_name)

        # Filtering values into features or targets
        if column_name in feature_columns:
            xs.append(encoded_values)
            feature_names.extend(tmp_names)
        if column_name in target_columns:
            ys.append(encoded_values)
            target_names.extend(tmp_names)

    # Concatenate features and targets
    x=None
    if xs:
        x = torch.cat(xs, dim=-1)
    target=None
    if ys:
        target = torch.cat(ys, dim=-1)

    
    index_name_map = {i:name for i, name in enumerate(names)}

    return x, target, index_name_map, feature_names, target_names

def load_relationship_parquet(path, 
                            feature_columns=[], 
                            target_columns=[],
                            custom_encoders={}, 
                            filter={}):
    
    all_columns=feature_columns+target_columns

    # Getting field metadata for all columns
    field_metadata=get_parquet_field_metadata(path,columns=all_columns)
    
    edge_name=os.path.basename(path).split('.')[0]
    src_name,edge_type,dst_name=edge_name.split('-')
    src_column_name=f'{src_name}-START_ID'
    dst_column_name=f'{dst_name}-END_ID'
    type_column_name='TYPE'

    # Reading all columns into single dataframe
    df = pd.read_parquet(path)

    edge_index_df=df[[src_column_name,dst_column_name]]
    df=df.drop(columns=[src_column_name,dst_column_name,type_column_name])

    column_names=list(df.columns)

    logger.info(f"Dataframe shape: {df.shape}")
    logger.info(f"Column names: {column_names}")

    # Ensure all columns have no NaN values, otherwise drop them
    for col in column_names:
        df=df.dropna(subset=[col])

    logger.info(f"Dataframe shape after removing NaN values: {df.shape}")

    # Apply data filter to the nodes
    if filter != {}:
        for key, value in filter.items():
            for name in column_names:
                if key in name:
                    column = name
                    break
            min_value, max_value = value
            df = df[(df[column] >= min_value) & (df[column] <= max_value)]

    logger.info(f"Dataframe shape after applying filter: {df.shape}")

    # Applying encoders to the nodes
    xs=[]
    ys=[]
    feature_names=[]
    target_names=[]
    for column_name in column_names:
        tmp_names=[]
        # Applying custom encoder if provided, 
        # otherwise use default encoder inside parquet feild metadata
        if column_name in custom_encoders:
            if isinstance(custom_encoders[column_name],str):
                encoder=eval(custom_encoders[column_name])
            else:
                encoder=custom_encoders[column_name]
        else:
            encoder=eval(field_metadata[column_name]['encoder'])

        tmp_values=encoder(df[column_name])

        # Getting feature names from encoder. Some encoders return multiple feature 
        # columns due to the nature of the encoder
        encoder_feature_names=None
        if hasattr(encoder,'column_names'):
            encoder_feature_names=encoder.column_names

        # Generating feature names
        if encoder_feature_names:
            for feature_name in encoder_feature_names:
                tmp_names.append(f"{column_name}_{feature_name}")
        elif tmp_values.shape[1]>1:
            for i in range(tmp_values.shape[1]):
                tmp_names.append(f"{column_name}_{i}")
        else:
            tmp_names.append(column_name)

        # Filtering values into features or targets
        if column_name in feature_columns:
            xs.append(tmp_values)
            feature_names.extend(tmp_names)
        if column_name in target_columns:
            ys.append(tmp_values)
            target_names.extend(tmp_names)

    # Concatenate features and targets
    edge_attr=None
    if xs:
        edge_attr = torch.cat(xs, dim=-1)
    target=None
    if ys:
        target = torch.cat(ys, dim=-1)
    edge_index=torch.from_numpy(edge_index_df.values).T

    # Create maping from original index to unqique index after removing posssible nan values
    index_name_mapping = {index: i for i, index in enumerate(df.index.unique())}

    return edge_index, edge_attr, target, index_name_mapping, feature_names, target_names

class DataGenerator:
    def __init__(self):
        self.hetero_data = HeteroData()
        self.node_id_mappings={}

        logger.info(f"Initializing DataGenerator")
        self._homo_data = None

    @property
    def homo_data(self):
        return self._homo_data
    @homo_data.getter
    def homo_data(self):
        try:
            return self.hetero_data.to_homogeneous()
        except Exception as e:
            raise ValueError(f"Make sure to only upload a the nodes have the same amount of features: {e}")
    @homo_data.setter
    def homo_data(self, value):
        self._homo_data = value

    def add_node_type(self, node_path, 
                    feature_columns=[], 
                    target_columns=[],
                    custom_encoders={}, 
                    filter={}):
        logger.info(f"Adding node type: {node_path}")


        node_name=os.path.basename(node_path).split('.')[0]

        x,target,index_name_map,feature_names,target_names=load_node_parquet(node_path, 
                                                                    feature_columns=feature_columns, 
                                                                    target_columns=target_columns,
                                                                    custom_encoders=custom_encoders,
                                                                    filter=filter)
        
        logger.info(f"{node_name} feature shape: {x.shape}")
        logger.info(f"{node_name} feature names: {len(feature_names)}")
        
        # logger.info(f"{node_name} index name map: {feature_names}")
        logger.info(f"{node_name} target name map: {target_names}")


        self.hetero_data[node_name].node_id=torch.arange(len(index_name_map))
        self.hetero_data[node_name].names=list(index_name_map.values())
        if x is not None:
            self.hetero_data[node_name].x = x
            self.hetero_data[node_name].feature_names=feature_names
        else:
            self.hetero_data[node_name].num_nodes=len(index_name_map)

        if target is not None:
            
            self.hetero_data[node_name].y_label_name=target_columns
            out_channels=target.shape[1]
            self.hetero_data[node_name].out_channels = out_channels

            if out_channels==1:
                self.hetero_data[node_name].y=target
                self.hetero_data[node_name].y_names=target_names
            else:
                self.hetero_data[node_name].y=torch.argmax(target, dim=1)

            logger.info(f"{node_name} target shape: {target.shape}")
            logger.info(f"{node_name} out channels: {out_channels}")
        
        logger.info(f"Node {node_name} added to the graph")
        
        self.node_id_mappings.update({node_name:index_name_map})

        return index_name_map

    def add_edge_type(self, edge_path,
                feature_columns=[], 
                target_columns=[],
                custom_encoders={}, 
                filter={},
                undirected=True):
        
        edge_name=os.path.basename(edge_path).split('.')[0]

        src_name,edge_type,dst_name=edge_name.split('-')
        
        graph_dir=os.path.dirname(os.path.dirname(edge_path))
        node_dir=os.path.join(graph_dir,'nodes')

        logger.info(f"Edge name | {edge_name}")
        logger.info(f"Edge type | {edge_type}")
        logger.info(f"Edge src name | {src_name}")
        logger.info(f"Edge dst name | {dst_name}")
        logger.info(f"Graph dir | {graph_dir}")
        logger.info(f"Node dir | {node_dir}")

        edge_index, edge_attr, target, index_name_map, feature_names, target_names = load_relationship_parquet(edge_path, 
                                                                                            feature_columns=feature_columns, 
                                                                                            target_columns=target_columns,
                                                                                            custom_encoders=custom_encoders, 
                                                                                            filter=filter)
        
        logger.info(f"Edge index shape: {edge_index.shape}")
        logger.info(f"Edge attr shape: {edge_attr.shape}")
        logger.info(f"Index name map: {index_name_map}")
        logger.info(f"Feature names: {feature_names}")
        logger.info(f"Target names: {target_names}")

        
        self.hetero_data[src_name,edge_type,dst_name].edge_index=edge_index
        self.hetero_data[src_name,edge_type,dst_name].edge_attr=edge_attr
        self.hetero_data[src_name,edge_type,dst_name].property_names=feature_names

        if target is not None:
            logger.info(f"Target shape: {target.shape}")
            self.hetero_data[src_name,edge_type,dst_name].y_label_name=target_names
            out_channels=target.shape[1]
            self.hetero_data[src_name,edge_type,dst_name].out_channels=out_channels

            if out_channels==1:
                self.hetero_data[src_name,edge_type,dst_name].y=target
            else:
                self.hetero_data[src_name,edge_type,dst_name].y=torch.argmax(target, dim=1)


        if undirected:
            
            row, col = edge_index
            rev_edge_index = torch.stack([col, row], dim=0)
            self.hetero_data[dst_name,f'rev_{edge_type}',src_name].edge_index=rev_edge_index
            self.hetero_data[dst_name,f'rev_{edge_type}',src_name].edge_attr=edge_attr

            if target is not None:
                self.hetero_data[dst_name,f'rev_{edge_type}',src_name].y_label_name=target_names
                out_channels=target.shape[1]
                self.hetero_data[dst_name,f'rev_{edge_type}',src_name].out_channels = out_channels


                if out_channels==1:
                    self.hetero_data[src_name,edge_type,dst_name].y=target
                else:
                    self.hetero_data[src_name,edge_type,dst_name].y=torch.argmax(target, dim=1)

        logger.info(f"Adding {edge_type} edge | {src_name} -> {dst_name}")
        
        
        logger.info(f"undirected: {undirected}")

    def to_homogeneous(self):
        logger.info(f"Converting to homogeneous graph")
        self.homo_data=self.hetero_data.to_homogeneous()

    def save_graph(self, filepath, use_buffer=False, homogeneous=False):
        file_type=filepath.split('.')[-1]

        if homogeneous==True:
            data=self.homo_data
            logger.info(f"Saving homogeneous graph")
        else:
            data=self.hetero_data
            logger.info(f"Saving heterogeneous graph")

        if file_type!='pt':
            raise ValueError("Only .pt files are supported")
        
        if use_buffer==True:
            buffer = io.BytesIO()
            torch.save(data, buffer)
        else:
            torch.save(data, filepath)

    def load_graph(self, filepath, use_buffer=False, homogeneous=False):
        file_type=filepath.split('.')[-1]
        if file_type!='pt':
            raise ValueError("Only .pt files are supported")
        
        
        if use_buffer==True:
            with open(filepath, 'rb') as f:
                buffer = io.BytesIO(f.read())
            data=torch.load(buffer, weights_only=False)
        else:
            data=torch.load(filepath, weights_only=False)

        if homogeneous==True:
            self.homo_data=data
            logger.info(f"Saving homogeneous graph")
        else:
            self.hetero_data=data
            logger.info(f"Saving heterogeneous graph")
        
        return data
    

if __name__ == "__main__":
    
    import pandas as pd
    import os
    material_graph=GraphManager(skip_init=False)
    graph_dir = material_graph.graph_dir
    nodes_dir = material_graph.node_dir
    relationship_dir = material_graph.relationship_dir
 

    node_names=material_graph.list_nodes()
    relationship_names=material_graph.list_relationships()

    node_files=material_graph.get_node_filepaths()
    relationship_files=material_graph.get_relationship_filepaths()

    print('-'*100)
    print('Nodes')
    print('-'*100)
    for i,node_file in enumerate(node_files):
        print(i,node_file)
    print('-'*100)
    print('Relationships')
    print('-'*100)
    for i,relationship_file in enumerate(relationship_files):
        print(i,relationship_file)
    print('-'*100)


    node_path=node_files[2]
    edge_path=relationship_files[3]

    # node_path=node_files[5]
    # df=pd.read_parquet(node_path)

    node_path=node_files[0]
    # df=pd.read_parquet(node_path)
    # # for x in df.columns:
    # #     print(f"'{x}',")
    # df.to_csv(node_path.replace('.parquet','.csv'))

    # df=pd.read_parquet(edge_path)
    # df.to_csv(edge_path.replace('.parquet','.csv'))

    ##############################################################################################################################
    # Example of using DataGenerator
    ##############################################################################################################################

    node_properties={
        'CHEMENV':[
            'coordination',
            'name',
        ],
        'ELEMENT':[
            'abundance_universe',
            'abundance_solar',
            'abundance_meteor',
            'abundance_crust',
            'abundance_ocean',
            'abundance_human',
            'atomic_mass',
            'atomic_number',
            'block',
            # 'boiling_point',
            'critical_pressure',
            'critical_temperature',
            'density_stp',
            'electron_affinity',
            'electronegativity_pauling',
            'extended_group',
            'heat_specific',
            'heat_vaporization',
            'heat_fusion',
            'heat_molar',
            'magnetic_susceptibility_mass',
            'magnetic_susceptibility_molar',
            'magnetic_susceptibility_volume',
            'melting_point',
            'molar_volume',
            'neutron_cross_section',
            'neutron_mass_absorption',
            'period',
            'phase',
            'radius_calculated',
            'radius_empirical',
            'radius_covalent',
            'radius_vanderwaals',
            'refractive_index',
            'speed_of_sound',
            # 'valence_electrons',
            'conductivity_electric',
            'electrical_resistivity',
            'modulus_bulk',
            'modulus_shear',
            'modulus_young',
            'poisson_ratio',
            'coefficient_of_linear_thermal_expansion',
            'hardness_vickers',
            'hardness_brinell',
            'hardness_mohs',
            'superconduction_temperature',
            'is_actinoid',
            'is_alkali',
            'is_alkaline',
            'is_chalcogen',
            'is_halogen',
            'is_lanthanoid',
            'is_metal',
            'is_metalloid',
            'is_noble_gas',
            'is_post_transition_metal',
            'is_quadrupolar',
            'is_rare_earth_metal',
            'experimental_oxidation_states',
            'name',
            'type',
        ],
        'MATERIAL':[
            'nsites',
            'nelements',
            'volume',
            'density',
            'density_atomic',
            'crystal_system',
            # 'space_group',
            # 'point_group',
            'a',
            'b',
            'c',
            'alpha',
            'beta',
            'gamma',
            'unit_cell_volume',
            # 'energy_per_atom',
            # 'formation_energy_per_atom',
            # 'energy_above_hull',
            # 'band_gap',
            # 'cbm',
            # 'vbm',
            # 'efermi',
            # 'is_stable',
            # 'is_gap_direct',
            # 'is_metal',
            # 'is_magnetic',
            # 'ordering',
            # 'total_magnetization',
            # 'total_magnetization_normalized_vol',
            # 'num_magnetic_sites',
            # 'num_unique_magnetic_sites',
            # 'e_total',
            # 'e_ionic',
            # 'e_electronic',
            # 'sine_coulomb_matrix',
            # 'element_fraction',
            # 'element_property',
            # 'xrd_pattern',
            # 'uncorrected_energy_per_atom',
            # 'equilibrium_reaction_energy_per_atom',
            # 'n',
            # 'e_ij_max',
            # 'weighted_surface_energy_EV_PER_ANG2',
            # 'weighted_surface_energy',
            # 'weighted_work_function',
            # 'surface_anisotropy',
            # 'shape_factor',
            # 'elasticity-k_vrh',
            # 'elasticity-k_reuss',
            # 'elasticity-k_voigt',
            # 'elasticity-g_vrh',
            # 'elasticity-g_reuss',
            # 'elasticity-g_voigt',
            # 'elasticity-sound_velocity_transverse',
            # 'elasticity-sound_velocity_longitudinal',
            # 'elasticity-sound_velocity_total',
            # 'elasticity-sound_velocity_acoustic',
            # 'elasticity-sound_velocity_optical',
            # 'elasticity-thermal_conductivity_clarke',
            # 'elasticity-thermal_conductivity_cahill',
            # 'elasticity-young_modulus',
            # 'elasticity-universal_anisotropy',
            # 'elasticity-homogeneous_poisson',
            # 'elasticity-debye_temperature',
            # 'elasticity-state',
            'name',
        ]
    }


    relationship_properties={
        'ELEMENT-CAN_OCCUR-CHEMENV':[
            'weight',
            ],

        'ELEMENT_GROUP_PERIOD_CONNECTS_ELEMENT':[
            'weight',
            ]
        }

    # generator=DataGenerator()
    # # generator.add_node_type(node_path=node_files[0], 
    # #                         feature_columns=node_properties['CHEMENV'],
    # #                         target_columns=[])
    
    # generator.add_node_type(node_path=node_files[2], 
    #                         feature_columns=node_properties['ELEMENT'],
    #                         target_columns=[])

    # # # # generator.add_node_type(node_path=node_path, 
    # # # #                         feature_columns=node_properties['MATERIAL'],
    # # # #                         target_columns=['elasticity-k_vrh'])
    
    # generator.add_edge_type(edge_path=edge_path,
    #                     feature_columns=relationship_properties['ELEMENT_GROUP_PERIOD_CONNECTS_ELEMENT'], 
    #                     # target_columns=['weight'],
    #                     # custom_encoders={}, 
    #                     # node_filter={},
    #                     undirected=True)
    

    # # print(generator.data)

    # # generator.save_graph(filepath=os.path.join('data','raw','main.pt'))

    # generator.load_graph(filepath=os.path.join('data','raw','main.pt'))

    # print(generator.data)

    ##############################################################################################################################
    # Creating graph node embeddings
    ##############################################################################################################################

    generator=DataGenerator()
    generator.add_node_type(node_path=node_files[2], 
                            feature_columns=node_properties['ELEMENT'],
                            target_columns=[])
    generator.add_edge_type(edge_path=relationship_files[8],
                        feature_columns=relationship_properties['ELEMENT_GROUP_PERIOD_CONNECTS_ELEMENT'], 
                        # target_columns=['weight'],
                        # custom_encoders={}, 
                        # node_filter={},
                        undirected=True)
    print(generator.hetero_data)
    data=generator.homo_data
    print(data)


    

    # generator.save_graph(filepath=os.path.join('data','raw','main.pt'))

    # generator.load_graph(filepath=os.path.join('data','raw','main.pt'))

    # print(generator.data)   

