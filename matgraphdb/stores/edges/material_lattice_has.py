import logging

import pyarrow as pa
import pyarrow.compute as pc
from parquetdb.utils import pyarrow_utils

from matgraphdb.stores.edge_store import EdgeStore

logger = logging.getLogger(__name__)


class MaterialLatticeHasEdges(EdgeStore):
    def __init__(self, storage_path, material_store_path, lattice_store_path):
        super().__init__(storage_path,
                         node_store_path_1=material_store_path, 
                         node_store_path_2=lattice_store_path)

    def post_initialize(self, **kwargs):
        try:
            
            connection_name= 'has'
            self.edge_type = f'{self.node_type_1}_{connection_name}_{self.node_type_2}'
            
            material_store = self.node_store_1
            lattice_store = self.node_store_2

            
            material_table = material_store.read_nodes(columns=['id','core.material_id'])
            lattice_table = lattice_store.read_nodes(columns=['material_node_id'])
            
            material_table=material_table.rename_columns({'id':'source_id', 'core.material_id':'material_id'})
            material_table=material_table.append_column('source_type', pa.array(['material']*material_table.num_rows))

            lattice_table=lattice_table.append_column('target_id', lattice_table['material_node_id'].combine_chunks())
            lattice_table=lattice_table.append_column('target_type', pa.array(['material_lattice']*lattice_table.num_rows))

            edge_table = pyarrow_utils.join_tables(material_table, lattice_table, 
                                      left_keys=['source_id'], right_keys=['material_node_id'],
                                      join_type='left outer')
            
            edge_table=edge_table.append_column('weight', pa.array([1.0]*edge_table.num_rows))
        
            logger.debug(f"Created {self.edge_type} edges. Shape: {edge_table.shape}")
        except Exception as e:
            logger.exception(f"Error creating {self.edge_type} relationships: {e}")
            raise e

        return edge_table