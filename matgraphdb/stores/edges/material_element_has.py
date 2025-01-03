import logging

import pyarrow as pa
import pyarrow.compute as pc
from parquetdb import ParquetDB
from parquetdb.utils import pyarrow_utils

from matgraphdb.stores.edge_store import EdgeStore

logger = logging.getLogger(__name__)


class MaterialElementHasEdges(EdgeStore):
    def __init__(self, storage_path, material_store_path, element_store_path):
        super().__init__(storage_path,
                         node_store_path_1=material_store_path, 
                         node_store_path_2=element_store_path)

    def post_initialize(self, **kwargs):
        try:
            
            connection_name= 'has'
            self.edge_type = f'{self.node_type_1}_{connection_name}_{self.node_type_2}'
            
            material_store = self.node_store_1
            element_store = self.node_store_2

            
            material_table = material_store.read_nodes(columns=['id','core.material_id','core.elements'])
            element_table = element_store.read_nodes(columns=['id','symbol'])
            
            material_table=material_table.rename_columns({'id':'source_id', 'core.material_id':'material_name'})
            material_table=material_table.append_column('source_type', pa.array(['material']*material_table.num_rows))
            
            element_table=element_table.rename_columns({'id':'target_id'})
            element_table=element_table.append_column('target_type', pa.array(['elements']*element_table.num_rows))

            material_df=material_table.to_pandas()
            element_df=element_table.to_pandas()
            element_target_id_map={row['symbol']:row['target_id'] for _, row in element_df.iterrows()}
            
            table_dict={
                'source_id':[],
                'source_type':[],
                'target_id':[],
                'target_type':[],
                'name':[],
                'weight':[]
            }
            
            for _, row in material_df.iterrows():
                elements = row['core.elements']
                source_id= row['source_id']
                material_name=row['material_name']
                if elements is None:
                    continue

                # Append the material name for each element in the species list
                for element in elements:
                    
                    target_id=element_target_id_map[element]
                    table_dict['source_id'].append(source_id)
                    table_dict['source_type'].append('material')
                    table_dict['target_id'].append(target_id)
                    table_dict['target_type'].append('element')
                    
                    name=f'{material_name}_{connection_name}_{element}'
                    table_dict['name'].append(name)
                    table_dict['weight'].append(1.0)
            
            edge_table=ParquetDB.construct_table(table_dict)
            
            self.name_column = 'name'
            
            
            
            logger.debug(f"Created {self.edge_type} edges. Shape: {edge_table.shape}")
        except Exception as e:
            logger.exception(f"Error creating {self.edge_type} relationships: {e}")
            raise e

        return edge_table