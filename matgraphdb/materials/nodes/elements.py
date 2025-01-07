import logging
import os
import warnings

import pandas as pd

from matgraphdb import PKG_DIR
from matgraphdb.core import NodeStore

logger = logging.getLogger(__name__)

BASE_ELEMENT_FILE = os.path.join(PKG_DIR, 'utils', 'chem_utils', 'resources','imputed_periodic_table_values.parquet')

class ElementNodes(NodeStore):
    def __init__(self, storage_path: str, base_element_file=BASE_ELEMENT_FILE):
        logger.debug(f"Initializing ElementNodes with storage path: {storage_path}")
        super().__init__(storage_path=storage_path, initialize_kwargs={'base_element_file':base_element_file})
        
    def initialize(self, base_element_file='imputed_periodic_table_values.parquet'):
        self.name_column = 'symbol'
        
        logger.info(f"Initializing element nodes from {base_element_file}")
        # Suppress warnings during node creation
        warnings.filterwarnings("ignore", category=UserWarning)

        try:
            file_ext = os.path.splitext(base_element_file)[-1][1:]
            logger.debug(f"File extension: {file_ext}")
            if file_ext == 'parquet':
                df = pd.read_parquet(os.path.join(PKG_DIR, 'utils', base_element_file))
            elif file_ext == 'csv':
                df = pd.read_csv(os.path.join(PKG_DIR, 'utils', base_element_file), index_col=0)
            else:
                raise ValueError(f"base_element_file must be a parquet or csv file")
            logger.debug(f"Read element dataframe shape {df.shape}")
            
            df['oxidation_states']=df['oxidation_states'].apply(lambda x: x.replace(']', '').replace('[', ''))
            df['oxidation_states']=df['oxidation_states'].apply(lambda x: ','.join(x.split()) )
            df['oxidation_states']=df['oxidation_states'].apply(lambda x: eval('['+x+']') )
            df['experimental_oxidation_states']=df['experimental_oxidation_states'].apply(lambda x: eval(x) )
            df['ionization_energies']=df['ionization_energies'].apply(lambda x: eval(x) )
            
        except Exception as e:
            logger.error(f"Error reading element CSV file: {e}")
            return None

        return df
