import io
import os

import pandas as pd
import pyarrow.parquet as pq
import torch
from torch_geometric.data import HeteroData

from matgraphdb.graph_kit.pyg.encoders import *
from matgraphdb.utils import get_child_logger

logger=get_child_logger(__name__, console_out=False, log_level='debug')

def get_parquet_field_metadata(path, columns=None):
    """
    Retrieves the metadata for each field (column) in a Parquet file and returns it in a dictionary format.
    
    Args:
        path (str): The file path to the Parquet file.
        columns (list, optional): A list of column names to filter and return metadata for. If None, metadata for all fields is returned.
        
    Returns:
        dict: A dictionary where keys are the column names, and values are dictionaries containing metadata for each column. 
              The metadata is extracted from the Arrow schema in the Parquet file and includes key-value pairs converted to UTF-8 strings.
              
    Example:
        field_metadata = get_parquet_field_metadata('data.parquet', columns=['column1', 'column2'])
    """
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

def load_node_parquet(path, feature_columns=[], target_columns=[], custom_encoders={}, filter={}, keep_nan=False):
    """
    Loads a Parquet file containing node data and applies custom encoders and filters, returning feature and target tensors 
    along with a mapping of node names to indices.

    Args:
        path (str): The file path to the Parquet file.
        feature_columns (list, optional): List of column names to use as features. Defaults to an empty list.
        target_columns (list, optional): List of column names to use as targets. Defaults to an empty list.
        custom_encoders (dict, optional): A dictionary where keys are column names and values are custom encoders to apply to those columns.
                                          If not provided, the default encoder from the Parquet metadata will be used.
        filter (dict, optional): A dictionary where keys are column names and values are tuples specifying the (min, max) range for filtering rows.
        keep_nan (bool, optional): If False, rows with NaN values will be dropped. Defaults to False.
        
    Returns:
        tuple: A tuple containing:
               - x (torch.Tensor): The feature tensor concatenated from all feature columns.
               - target (torch.Tensor): The target tensor concatenated from all target columns.
               - index_name_map (dict): A mapping of the original index to the name of the node.
               - feature_names (list): A list of feature column names, including any columns expanded by the encoders.
               - target_names (list): A list of target column names, including any columns expanded by the encoders.

    Example:
        x, target, index_name_map, feature_names, target_names = load_node_parquet('nodes.parquet', feature_columns=['f1'], target_columns=['t1'])
    """
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

    # Ensure all columns have no NaN values, otherwise drop them
    if not keep_nan:
        df.dropna(subset=df.columns, inplace=True)

    logger.info(f"Dataframe shape after removing NaN values: {df.shape}")

    # Apply data filter to the nodes
    for key, (min_value, max_value) in filter.items():
        df = df[(df[key] >= min_value) & (df[key] <= max_value)]

    logger.info(f"Dataframe shape after applying node filter: {df.shape}")
    logger.info(f"Dataframe shape: {df.shape}")
    logger.info(f"Column names: {column_names}")

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

    # This maps the original index to the name of the node. 
    # This is needed since we filter out nodes that contain NaN.
    index_name_map = {index: names[index] for i, index in enumerate(df.index.unique())}
    return x, target, index_name_map, feature_names, target_names

def load_relationship_parquet(path, node_id_mappings,
                            feature_columns=[], 
                            target_columns=[],
                            custom_encoders={}, 
                            filter={}):
    """
    Loads a Parquet file containing relationship data (edges between nodes), applies custom encoders and filters, 
    and returns edge indices, edge attributes, and target tensors along with a mapping of edge indices.

    Args:
        path (str): The file path to the Parquet file.
        node_id_mappings (dict): A dictionary where keys are node types (e.g., 'src', 'dst') and values are mappings of node indices to IDs.
        feature_columns (list, optional): List of column names to use as edge features. Defaults to an empty list.
        target_columns (list, optional): List of column names to use as edge targets. Defaults to an empty list.
        custom_encoders (dict, optional): A dictionary where keys are column names and values are custom encoders to apply to those columns.
                                          If not provided, the default encoder from the Parquet metadata will be used.
        filter (dict, optional): A dictionary where keys are column names and values are tuples specifying the (min, max) range for filtering rows.
        
    Returns:
        tuple: A tuple containing:
               - edge_index (torch.Tensor): The edge index tensor with source and destination node indices.
               - edge_attr (torch.Tensor): The edge attributes tensor concatenated from all feature columns.
               - target (torch.Tensor): The target tensor concatenated from all target columns.
               - index_name_mapping (dict): A mapping of the original edge index to the filtered edge index.
               - feature_names (list): A list of feature column names, including any columns expanded by the encoders.
               - target_names (list): A list of target column names, including any columns expanded by the encoders.

    Example:
        edge_index, edge_attr, target, index_name_mapping, feature_names, target_names = load_relationship_parquet(
            'edges.parquet', node_id_mappings={'src': {0: 'A'}, 'dst': {1: 'B'}}, feature_columns=['f1'])
    """
    
    all_columns=feature_columns+target_columns

    # Getting field metadata for all columns
    field_metadata=get_parquet_field_metadata(path,columns=all_columns)
    
    # Getting relationship information
    edge_name=os.path.basename(path).split('.')[0]
    src_name,edge_type,dst_name=edge_name.split('-')
    src_column_name=f'{src_name}-START_ID'
    dst_column_name=f'{dst_name}-END_ID'
    src_index_name_mapping=node_id_mappings[src_name]
    dst_index_name_mapping=node_id_mappings[dst_name]

    # This maps the reduced index to the original index. 
    # Again this is because we filter out nodes that contain NaN.
    src_index_translation={index:reduced_index for reduced_index,index in enumerate(src_index_name_mapping.keys())}
    dst_index_translation={index:reduced_index for reduced_index,index in enumerate(dst_index_name_mapping.keys())}
    
    type_column_name='TYPE'

    # Reading all columns into single dataframe
    df = pd.read_parquet(path)

    edges_in_graph_mask=df[src_column_name].isin(src_index_name_mapping.keys()) & df[dst_column_name].isin(dst_index_name_mapping.keys())
    df = pd.DataFrame(df[edges_in_graph_mask].to_dict())

    edge_index=None
    edge_attr = None
    df['src_mapped'] = df[src_column_name].map(src_index_translation)
    df['dst_mapped'] = df[dst_column_name].map(dst_index_translation)

    edge_index = torch.tensor([df['src_mapped'].tolist(),
                               df['dst_mapped'].tolist()])

    df=df.drop(columns=[src_column_name,dst_column_name,type_column_name,'src_mapped','dst_mapped'])

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


    # Create maping from original index to unqique index after removing posssible nan values
    index_name_mapping = {index: i for i, index in enumerate(df.index.unique())}

    return edge_index, edge_attr, target, index_name_mapping, feature_names, target_names

class DataGenerator:
    """
    A class used to generate and manage heterogeneous and homogeneous graph data for machine learning models.
    This class handles the loading of nodes and edges, conversion between heterogeneous and homogeneous graph formats,
    and saving/loading of graph data.

    Attributes
    ----------
    hetero_data : HeteroData
        Stores the heterogeneous graph data, where each node and edge type can have its own features and labels.
    node_id_mappings : dict
        Maps node names to their index IDs, helping to manage relationships between nodes when edges are added.
    _homo_data : torch_geometric.data.Data, optional
        Stores the homogeneous graph data, if converted from the heterogeneous data.

    Methods
    -------
    homo_data()
        Getter property that converts the heterogeneous graph to homogeneous format if requested.
    add_node_type(node_path, feature_columns=[], target_columns=[], custom_encoders={}, filter={}, keep_nan=False)
        Adds a node type (with optional features and targets) to the heterogeneous graph.
    add_edge_type(edge_path, feature_columns=[], target_columns=[], custom_encoders={}, filter={}, undirected=True)
        Adds an edge type between two nodes in the heterogeneous graph, with optional edge features and labels.
    to_homogeneous()
        Converts the heterogeneous graph to a homogeneous format, where all node and edge types are merged.
    save_graph(filepath, use_buffer=False, homogeneous=False)
        Saves the graph data (either homogeneous or heterogeneous) to a specified file in .pt format.
    load_graph(filepath, use_buffer=False, homogeneous=False)
        Loads graph data (either homogeneous or heterogeneous) from a .pt file.
    """
    def __init__(self):
        """
        Initializes the DataGenerator class.

        Sets up an empty heterogeneous graph (`hetero_data`) and initializes mappings for node IDs.
        Also initializes `_homo_data` to store the homogeneous graph, if needed.
        """
        self.hetero_data = HeteroData()

        # This is need to map if nodes are filtered out.
        self.node_id_mappings={}

        logger.info(f"Initializing DataGenerator")
        self._homo_data = None

    @property
    def homo_data(self):
        
        return self._homo_data
    @homo_data.getter
    def homo_data(self):
        """
        Getter for the homogeneous graph data.

        Returns the homogeneous graph by converting the heterogeneous graph. Raises an exception if 
        the node features between types do not match.

        Returns
        -------
        torch_geometric.data.Data
            The homogeneous graph data.

        Raises
        ------
        ValueError
            If node features are inconsistent across types.
        """
        try:
            return self.hetero_data.to_homogeneous()
        except Exception as e:
            raise ValueError(f"Make sure to only upload a the nodes have the same amount of features: {e}")
    @homo_data.setter
    def homo_data(self, value):
        """
        Setter for the homogeneous graph data.

        Parameters
        ----------
        value : torch_geometric.data.Data
            The homogeneous graph data to be assigned.
        """
        self._homo_data = value

    def add_node_type(self, node_path, 
                    feature_columns=[], 
                    target_columns=[],
                    custom_encoders={}, 
                    filter={},
                    keep_nan=False):
        """
        Adds a new node type to the heterogeneous graph, loading features and targets from a Parquet file.

        Parameters
        ----------
        node_path : str
            Path to the Parquet file containing node data.
        feature_columns : list, optional
            List of column names to be used as features for the nodes.
        target_columns : list, optional
            List of column names to be used as target labels for the nodes.
        custom_encoders : dict, optional
            Custom encoders for specific feature columns.
        filter : dict, optional
            A dictionary of filters to apply when loading the node data.
        keep_nan : bool, optional
            Whether to keep NaN values in the data (default is False).

        Returns
        -------
        dict
            A mapping of node IDs to their original index names.

        Raises
        ------
        Exception
            If the node cannot be added due to an error in loading the data.
        """
        logger.info(f"Adding node type: {node_path}")


        node_name=os.path.basename(node_path).split('.')[0]

        x,target,index_name_map,feature_names,target_names=load_node_parquet(node_path, 
                                                                    feature_columns=feature_columns, 
                                                                    target_columns=target_columns,
                                                                    custom_encoders=custom_encoders,
                                                                    filter=filter,
                                                                    keep_nan=keep_nan)
        
        if x is not None:
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
        """
        Adds a new edge type between two nodes in the heterogeneous graph, with optional edge features and targets.

        Parameters
        ----------
        edge_path : str
            Path to the Parquet file containing edge data.
        feature_columns : list, optional
            List of column names to be used as features for the edges.
        target_columns : list, optional
            List of column names to be used as target labels for the edges.
        custom_encoders : dict, optional
            Custom encoders for specific feature columns.
        filter : dict, optional
            A dictionary of filters to apply when loading the edge data.
        undirected : bool, optional
            Whether to treat the edge as undirected and add a reverse edge (default is True).

        Raises
        ------
        Exception
            If the source or destination node types do not exist in the node ID mappings.
        """

        edge_name=os.path.basename(edge_path).split('.')[0]

        src_name,edge_type,dst_name=edge_name.split('-')

        if src_name not in self.node_id_mappings:
            raise Exception(f"Node {src_name} not found in node ID mappings. Call add_node_type first")

        if dst_name not in self.node_id_mappings:
            raise Exception(f"Node {dst_name} not found in node ID mappings. Call add_node_type first")
        
        graph_dir=os.path.dirname(os.path.dirname(edge_path))
        node_dir=os.path.join(graph_dir,'nodes')

        logger.info(f"Edge name | {edge_name}")
        logger.info(f"Edge type | {edge_type}")
        logger.info(f"Edge src name | {src_name}")
        logger.info(f"Edge dst name | {dst_name}")
        logger.info(f"Graph dir | {graph_dir}")
        logger.info(f"Node dir | {node_dir}")

        edge_index, edge_attr, target, index_name_map, feature_names, target_names = load_relationship_parquet(edge_path, 
                                                                                            self.node_id_mappings,
                                                                                            feature_columns=feature_columns, 
                                                                                            target_columns=target_columns,
                                                                                            custom_encoders=custom_encoders, 
                                                                                            filter=filter)
        
        logger.info(f"Edge index shape: {edge_index.shape}")
        if edge_attr is not None:
            logger.info(f"Edge attr shape: {edge_attr.shape}")
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
        """
        Converts the heterogeneous graph into a homogeneous graph, merging all node and edge types.

        Returns
        -------
        torch_geometric.data.Data
            The homogeneous graph data.
        """
        logger.info(f"Converting to homogeneous graph")
        self.homo_data=self.hetero_data.to_homogeneous()

    def save_graph(self, filepath, use_buffer=False, homogeneous=False):
        """
        Saves the graph data (either homogeneous or heterogeneous) to a specified file in .pt format.

        Parameters
        ----------
        filepath : str
            The path to save the graph data, which must have a `.pt` extension.
        use_buffer : bool, optional
            Whether to save the graph data to a buffer instead of directly to a file (default is False).
        homogeneous : bool, optional
            Whether to save the homogeneous graph data (default is False, saving the heterogeneous graph).

        Raises
        ------
        ValueError
            If the file type is not `.pt`.
        """
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
        """
        Loads graph data (either homogeneous or heterogeneous) from a .pt file.

        Parameters
        ----------
        filepath : str
            The path to load the graph data from, which must have a `.pt` extension.
        use_buffer : bool, optional
            Whether to load the graph data from a buffer instead of directly from a file (default is False).
        homogeneous : bool, optional
            Whether to load the homogeneous graph data (default is False, loading the heterogeneous graph).

        Returns
        -------
        torch_geometric.data.Data
            The loaded graph data.

        Raises
        ------
        ValueError
            If the file type is not `.pt`.
        """
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
    from matgraphdb.graph_kit.graph_manager import GraphManager
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
            # 'name',
        ]
    }


    relationship_properties={
        'ELEMENT-CAN_OCCUR-CHEMENV':[
            'weight',
            ],

        'ELEMENT_GROUP_PERIOD_CONNECTS_ELEMENT':[
            'weight',
            ],
        'MATERIAL-HAS-ELEMENT':[
            'weight',
            ]
        }

    generator=DataGenerator()
    generator.add_node_type(node_path=node_files[0], 
                            feature_columns=node_properties['CHEMENV'],
                            target_columns=[])
    
    generator.add_node_type(node_path=node_files[2], 
                            feature_columns=node_properties['ELEMENT'],
                            target_columns=[])

    generator.add_node_type(node_path=node_files[5], 
                            feature_columns=node_properties['MATERIAL'],
                            target_columns=['elasticity-k_vrh'],
                            filter={'elasticity-k_vrh':(0,300)}
                            )
    
    generator.add_edge_type(edge_path=edge_path,
                        feature_columns=relationship_properties['ELEMENT_GROUP_PERIOD_CONNECTS_ELEMENT'], 
                        # target_columns=['weight'],
                        # custom_encoders={}, 
                        # node_filter={},
                        undirected=True)
    generator.add_edge_type(edge_path='Z:/Research_Projects/crystal_generation_project/MatGraphDB/data/production/materials_project/graph_database/main/relationships/MATERIAL-HAS-ELEMENT.parquet',
                        feature_columns=relationship_properties['MATERIAL-HAS-ELEMENT'],
                        # target_columns=['weight'],
                        # custom_encoders={}, 
                        # node_filter={},
                        undirected=True)

    print(generator.hetero_data)

    print(generator.hetero_data['MATERIAL'].node_id)




    # # generator.save_graph(filepath=os.path.join('data','raw','main.pt'))

    # generator.load_graph(filepath=os.path.join('data','raw','main.pt'))

    # print(generator.data)

    ##############################################################################################################################
    # Creating graph node embeddings
    ##############################################################################################################################

    # generator=DataGenerator()
    # generator.add_node_type(node_path=node_files[2], 
    #                         feature_columns=node_properties['ELEMENT'],
    #                         target_columns=[])
    # generator.add_edge_type(edge_path=relationship_files[8],
    #                     feature_columns=relationship_properties['ELEMENT_GROUP_PERIOD_CONNECTS_ELEMENT'], 
    #                     # target_columns=['weight'],
    #                     # custom_encoders={}, 
    #                     # node_filter={},
    #                     undirected=True)
    # print(generator.hetero_data)
    # data=generator.homo_data
    # print(data)


    

    # generator.save_graph(filepath=os.path.join('data','raw','main.pt'))

    # generator.load_graph(filepath=os.path.join('data','raw','main.pt'))

    # print(generator.data)   

