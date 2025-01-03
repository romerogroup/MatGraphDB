import os
import warnings
import logging
from glob import glob

import pyarrow as pa
import pandas as pd
import numpy as np
import pyarrow.compute as pc
import pyarrow as pa

from matgraphdb.stores.edge_store import EdgeStore
from matgraphdb.stores.nodes import ElementNodes
from matgraphdb import PKG_DIR

from matgraphdb.utils.chem_utils.periodic import get_group_period_edge_index
from parquetdb import ParquetDB
from parquetdb.utils import pyarrow_utils
logger = logging.getLogger(__name__)


class MaterialSPGHasEdges(EdgeStore):
    def __init__(self, storage_path, material_store_path, spg_store_path):
        super().__init__(storage_path,
                         node_store_path_1=material_store_path, 
                         node_store_path_2=spg_store_path)

    def post_initialize(self, **kwargs):
        try:
            connection_name = 'has'
            self.edge_type = f'{self.node_type_1}_{connection_name}_{self.node_type_2}'
            
            material_store = self.node_store_1
            spg_store = self.node_store_2

            
            material_table = material_store.read_nodes(columns=['id','core.material_id','symmetry.number'])
            spg_table = spg_store.read_nodes(columns=['id','spg'])
            
            material_table=material_table.rename_columns({'id':'source_id', 'symmetry.number':'spg'})
            material_table=material_table.append_column('source_type', pa.array(['material']*material_table.num_rows))
            
            spg_table=spg_table.rename_columns({'id':'target_id'})
            spg_table=spg_table.append_column('target_type', pa.array(['spg']*spg_table.num_rows))

            
            edge_table = pyarrow_utils.join_tables(material_table, spg_table, 
                                      left_keys=['spg'], right_keys=['spg'],
                                      join_type='left outer')
            
            
            edge_table=edge_table.append_column('weight', pa.array([1.0]*edge_table.num_rows))

            names = pc.binary_join_element_wise(
                    pc.cast(edge_table['core.material_id'], pa.string()),
                    pc.cast(edge_table['spg'], pa.string()),
                    f'_{connection_name}_SpaceGroup')
            
            edge_table = edge_table.append_column('name', names)
            self.name_column = 'name'
            
            logger.debug(f"Created {self.edge_type} edges. Shape: {edge_table.shape}")
        except Exception as e:
            logger.exception(f"Error creating {self.edge_type} relationships: {e}")
            raise e

        return edge_table