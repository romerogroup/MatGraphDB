import logging

import pyarrow as pa
import pyarrow.compute as pc
from parquetdb import ParquetDB
from parquetdb.utils import pyarrow_utils

from matgraphdb.stores.edge_store import EdgeStore

logger = logging.getLogger(__name__)


class MaterialChemEnvContainsSiteEdges(EdgeStore):
    def __init__(self, storage_path, material_store_path, chemenv_store_path):
        super().__init__(storage_path,
                         node_store_path_1=material_store_path, 
                         node_store_path_2=chemenv_store_path)

    def post_initialize(self, **kwargs):
        try:
            
            connection_name= 'containsSite'
            self.edge_type = f'{self.node_type_1}_{connection_name}_{self.node_type_2}'
            
            material_store = self.node_store_1
            chemenv_store = self.node_store_2

            
            material_table = material_store.read_nodes(columns=['id','core.material_id','chemenv.coordination_environments_multi_weight'])
            chemenv_table = chemenv_store.read_nodes(columns=['id','mp_symbol'])

            material_table=material_table.rename_columns({'id':'source_id', 'core.material_id':'material_name'})
            material_table=material_table.append_column('source_type', pa.array(['material']*material_table.num_rows))
            
            chemenv_table=chemenv_table.rename_columns({'id':'target_id', 'mp_symbol':'chemenv_name'})
            chemenv_table=chemenv_table.append_column('target_type', pa.array(['chemenv']*chemenv_table.num_rows))

            material_df=material_table.to_pandas()
            chemenv_df=chemenv_table.to_pandas()
            chemenv_target_id_map={row['chemenv_name']:row['target_id'] for _, row in chemenv_df.iterrows()}
            
            table_dict={
                'source_id':[],
                'source_type':[],
                'target_id':[],
                'target_type':[],
                'name':[],
                'weight':[]
            }
            
            for _, row in material_df.iterrows():
                coord_envs = row['chemenv.coordination_environments_multi_weight']
                if coord_envs is None:
                    continue
                
                source_id= row['source_id']
                material_name=row['material_name']
                
                
                for coord_env in coord_envs:
                    try:
                        chemenv_name = coord_env[0]['ce_symbol']
                        target_id=chemenv_target_id_map[chemenv_name]
                    except:
                        continue
                    
                    
                    table_dict['source_id'].append(source_id)
                    table_dict['source_type'].append('material')
                    table_dict['target_id'].append(target_id)
                    table_dict['target_type'].append('chemenv')
                    
                    name=f'{material_name}_{connection_name}_{chemenv_name}'
                    table_dict['name'].append(name)
                    table_dict['weight'].append(1.0)
            
            edge_table=ParquetDB.construct_table(table_dict)
            
            self.name_column = 'name'
            
            
            
            logger.debug(f"Created {self.edge_type} edges. Shape: {edge_table.shape}")
        except Exception as e:
            logger.exception(f"Error creating {self.edge_type} relationships: {e}")
            raise e

        return edge_table