import logging

import pyarrow as pa
import pyarrow.compute as pc
from parquetdb.utils import pyarrow_utils

from matgraphdb.stores.edge_store import EdgeStore

logger = logging.getLogger(__name__)


class MaterialCrystalSystemHasEdges(EdgeStore):
    def __init__(self, storage_path, material_store_path, crystal_system_store_path):
        super().__init__(storage_path,
                         node_store_path_1=material_store_path, 
                         node_store_path_2=crystal_system_store_path)

    def post_initialize(self, **kwargs):
        try:
            
            connection_name= 'has'
            self.edge_type = f'{self.node_type_1}_{connection_name}_{self.node_type_2}'
            
            material_store = self.node_store_1
            crystal_system_store = self.node_store_2

            
            material_table = material_store.read_nodes(columns=['id','core.material_id','symmetry.crystal_system'])
            crystal_system_table = crystal_system_store.read_nodes(columns=['id','crystal_system'])
            
            material_table=material_table.rename_columns({'id':'source_id', 'symmetry.crystal_system':'crystal_system'})
            material_table=material_table.append_column('source_type', pa.array(['material']*material_table.num_rows))
            
            crystal_system_table=crystal_system_table.rename_columns({'id':'target_id'})
            crystal_system_table=crystal_system_table.append_column('target_type', pa.array(['crystal_system']*crystal_system_table.num_rows))

            
            edge_table = pyarrow_utils.join_tables(material_table, crystal_system_table, 
                                      left_keys=['crystal_system'], right_keys=['crystal_system'],
                                      join_type='left outer')
            
            
            edge_table=edge_table.append_column('weight', pa.array([1.0]*edge_table.num_rows))
            

            names = pc.binary_join_element_wise(
                    pc.cast(edge_table['core.material_id'], pa.string()),
                    pc.cast(edge_table['crystal_system'], pa.string()),
                    f'_{connection_name}_')
            
            edge_table = edge_table.append_column('name', names)
            
            self.name_column = 'name'
            
            
            
            logger.debug(f"Created {self.edge_type} edges. Shape: {edge_table.shape}")
        except Exception as e:
            logger.exception(f"Error creating {self.edge_type} relationships: {e}")
            raise e

        return edge_table