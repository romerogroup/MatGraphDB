import logging

import pyarrow as pa
import pyarrow.compute as pc
from parquetdb import ParquetDB
from parquetdb.utils import pyarrow_utils

from matgraphdb.stores.edge_store import EdgeStore

logger = logging.getLogger(__name__)


class ElementOxiStateCanOccurEdges(EdgeStore):
    def __init__(self, storage_path, element_store_path, oxiState_store_path):
        super().__init__(storage_path,
                         node_store_path_1=element_store_path, 
                         node_store_path_2=oxiState_store_path)

    def post_initialize(self, **kwargs):
        try:
            
            connection_name= 'canOccur'
            self.edge_type = f'{self.node_type_1}_{connection_name}_{self.node_type_2}'
            
            element_store = self.node_store_1
            oxiState_store = self.node_store_2

            element_table = element_store.read_nodes(columns=['id','experimental_oxidation_states','symbol'])
            oxiState_table = oxiState_store.read_nodes(columns=['id','oxidation_state','value'])
            
            # element_table=element_table.rename_columns({'id':'source_id'})
            element_table=element_table.append_column('source_type', pa.array(['element']*element_table.num_rows))
            
            # oxiState_table=oxiState_table.rename_columns({'id':'target_id'})
            oxiState_table=oxiState_table.append_column('target_type', pa.array(['oxiState']*oxiState_table.num_rows))

            element_df=element_table.to_pandas()
            oxiState_df=oxiState_table.to_pandas()
            table_dict={
                'source_id':[],
                'source_type':[],
                'target_id':[],
                'target_type':[],
                'name':[],
                'weight':[]
            }
            
            oxiState_id_map={}
            id_oxidationState_map={}
            for i,oxiState_row in oxiState_df.iterrows():
                oxiState_id_map[oxiState_row['value']]=oxiState_row['id']
                id_oxidationState_map[oxiState_row['id']]=oxiState_row['oxidation_state']
                
            for i,element_row in element_df.iterrows():
                exp_oxidation_states=element_row['experimental_oxidation_states']
                source_id=element_row['id']
                source_type='element'
                symbol=element_row['symbol']
                for exp_oxidation_state in exp_oxidation_states:
                    target_id=oxiState_id_map[exp_oxidation_state]
                    target_type='oxiState'
                    oxi_state_name=id_oxidationState_map[target_id]
                    
                    table_dict['source_id'].append(source_id)
                    table_dict['source_type'].append(source_type)
                    table_dict['target_id'].append(target_id)
                    table_dict['target_type'].append(target_type)
                    table_dict['weight'].append(1.0)
                    table_dict['name'].append(f'{symbol}_{connection_name}_{oxi_state_name}')

            edge_table = ParquetDB.construct_table(table_dict)
            
            self.name_column = 'name'
            
            
            
            # names = pc.binary_join_element_wise(
            #         pc.cast(edge_table['core.material_id'], pa.string()),
            #         pc.cast(edge_table['crystal_system'], pa.string()),
            #         f'_{connection_name}_')
            
            # edge_table = edge_table.append_column('name', names)
            
            # self.name_column = 'name'
            
            
            
            logger.debug(f"Created {self.edge_type} edges. Shape: {edge_table.shape}")
        except Exception as e:
            logger.exception(f"Error creating {self.edge_type} relationships: {e}")
            raise e

        return edge_table