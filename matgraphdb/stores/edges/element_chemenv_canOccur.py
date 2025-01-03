import logging

import pyarrow as pa
import pyarrow.compute as pc
from parquetdb import ParquetDB
from parquetdb.utils import pyarrow_utils

from matgraphdb.stores.edge_store import EdgeStore
from matgraphdb.stores.nodes.materials import MaterialNodes

logger = logging.getLogger(__name__)


class ElementChemEnvCanOccurRelationships(EdgeStore):
    def __init__(self, storage_path, element_store_path, chemenv_store_path, material_store_path):
        
        post_initialize_kwargs={
            'material_store_path':material_store_path
        }
        super().__init__(storage_path,
                         node_store_path_1=element_store_path, 
                         node_store_path_2=chemenv_store_path,
                         post_initialize_kwargs=post_initialize_kwargs)

    def post_initialize(self, material_store_path=None, **kwargs):
        try:
            
            connection_name= 'canOccur'
            self.edge_type = f'{self.node_type_1}_{connection_name}_{self.node_type_2}'
            
            element_store = self.node_store_1
            chemenv_store = self.node_store_2
            material_store = MaterialNodes(material_store_path)

            
            material_table = material_store.read_nodes(columns=['id','core.material_id','core.elements',
                                                                'chemenv.coordination_environments_multi_weight'])
            
            
            chemenv_table = chemenv_store.read_nodes(columns=['id','mp_symbol'])
            element_table = element_store.read_nodes(columns=['id','symbol'])

            chemenv_table=chemenv_table.rename_columns({'mp_symbol':'name'})
            chemenv_table=chemenv_table.append_column('target_type', pa.array(['chemenv']*chemenv_table.num_rows))
            
            element_table=element_table.rename_columns({'symbol':'name'})
            element_table=element_table.append_column('source_type', pa.array(['element']*element_table.num_rows))
            
            material_df=material_table.to_pandas()
            chemenv_df=chemenv_table.to_pandas()
            element_df=element_table.to_pandas()
            
            chemenv_target_id_map={row['name']:row['id'] for _, row in chemenv_df.iterrows()}
            element_target_id_map={row['name']:row['id'] for _, row in element_df.iterrows()}

            table_dict={
                'source_id':[],
                'source_type':[],
                'target_id':[],
                'target_type':[],
                'name':[],
            }
            
            for _, row in material_df.iterrows():
                coord_envs = row['chemenv.coordination_environments_multi_weight']
                
                if coord_envs is None:
                    continue
                
                elements = row['core.elements']
                
                for i, coord_env in enumerate(coord_envs):
                    try:
                        chemenv_name = coord_env[0]['ce_symbol']
                        element_name = elements[i]
                        
                        source_id=element_target_id_map[element_name]
                        target_id=chemenv_target_id_map[chemenv_name]
                    except:
                        continue
                    
                    
                    table_dict['source_id'].append(source_id)
                    table_dict['source_type'].append('element')
                    table_dict['target_id'].append(target_id)
                    table_dict['target_type'].append('chemenv')
                    
                    name=f'{element_name}_{connection_name}_{chemenv_name}'
                    table_dict['name'].append(name)
            
            edge_table=ParquetDB.construct_table(table_dict)
            
            self.name_column = 'name'
            
            
            
            logger.debug(f"Created {self.edge_type} edges. Shape: {edge_table.shape}")
        except Exception as e:
            logger.exception(f"Error creating {self.edge_type} relationships: {e}")
            raise e

        return edge_table