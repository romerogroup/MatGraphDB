# Check if encoings are present
import os
import json
from glob import glob

import pandas as pd
import pymatgen.core as pmat

from matgraphdb.database.neo4j.node_types import (ELEMENTS, MAGNETIC_STATES, CRYSTAL_SYSTEMS, CHEMENV_NAMES,
                                                  MATERIAL_FILES, CHEMENV_ELEMENT_NAMES, SPG_NAMES)
from matgraphdb.database.json.utils import PROPERTY_NAMES
from matgraphdb.utils import  GLOBAL_PROP_FILE, NODE_DIR, LOGGER, ENCODING_DIR



node_dict={}
if os.path.exists(ENCODING_DIR):
    encoding_files=glob(os.path.join(ENCODING_DIR,'*.csv'))
    for encoding_file in encoding_files:
        encoding_name=encoding_file.split(os.sep)[-1].split('.')[0]

        df=pd.read_csv(encoding_file,index_col=0)

        # Convert the dataframe values to a list of strings where the strings are the rows of the dataframe separated by a semicolon
        df = df.apply(lambda x: ';'.join(map(str, x)), axis=1)
        print(df.tolist())
        # node_dict.update({encoding_name: df.tolist()})