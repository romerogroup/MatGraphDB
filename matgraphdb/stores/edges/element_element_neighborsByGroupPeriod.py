import logging

import pyarrow as pa
import pandas as pd
import numpy as np
import pyarrow.compute as pc
from parquetdb import ParquetDB
from parquetdb.utils import pyarrow_utils


from matgraphdb.stores.edge_store import EdgeStore
from matgraphdb.utils.chem_utils.periodic import get_group_period_edge_index

logger = logging.getLogger(__name__)


class ElementElementNeighborsByGroupPeriodEdges(EdgeStore):
    def __init__(self, storage_path, element_store_path):
        super().__init__(storage_path,
                         node_store_path_1=element_store_path, 
                         node_store_path_2=element_store_path)

    def post_initialize(self, **kwargs):
        try:
            connection_name = 'neighborsByGroupPeriod'
            
            self.edge_type = f'{self.node_type_1}_{connection_name}_{self.node_type_2}'
            
            element_store = self.node_store_1
            table = element_store.read_nodes(columns=['atomic_number', 'extended_group', 'period', 'symbol'])
            element_df = table.to_pandas()
            
            # Getting group-period edge index
            edge_index = get_group_period_edge_index(element_df)

            # Creating the relationships dataframe
            df = pd.DataFrame(edge_index, columns=[f'source_id', f'target_id'])

            # Dropping rows with NaN values and casting to int64
            df = df.dropna().astype(np.int64)
        
            # Add source and target type columns
            df['source_type'] = 'elements'
            df['target_type'] = 'elements'
            df['weight'] = 1.0
            
            table=ParquetDB.construct_table(df)
            
            reduced_table=element_store.read(columns=['symbol','id','extended_group','period'])
            reduced_source_table=reduced_table.rename_columns({'symbol':'source_name', 'extended_group':'source_extended_group', 'period':'source_period'})
            reduced_target_table=reduced_table.rename_columns({'symbol':'target_name', 'extended_group':'target_extended_group', 'period':'target_period'})
            
            table=pyarrow_utils.join_tables(table, reduced_source_table, 
                                      left_keys=['source_id'], right_keys=['id'],
                                      join_type='left outer')
            
            table=pyarrow_utils.join_tables(table, reduced_target_table, 
                                      left_keys=['target_id'], right_keys=['id'],
                                      join_type='left outer')

            
            names = pc.binary_join_element_wise(
                    pc.cast(table['source_name'], pa.string()),
                    pc.cast(table['target_name'], pa.string()),
                    f'_{connection_name}_')
            
            table = table.append_column('name', names)
            self.name_column = 'name'
            
            logger.debug(f"Created {self.edge_type} edges. Shape: {table.shape}")
        except Exception as e:
            logger.exception(f"Error creating element-group-period relationships: {e}")
            raise e

        return table