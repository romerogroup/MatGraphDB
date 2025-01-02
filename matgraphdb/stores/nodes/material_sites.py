import os
import warnings
import logging
from glob import glob

import pandas as pd
import numpy as np
from matgraphdb.stores.node_store import NodeStore
from matgraphdb.stores.nodes.materials import MaterialNodes
import pyarrow.compute as pc
import pyarrow as pa
logger = logging.getLogger(__name__)

class MaterialSiteNodes(NodeStore):
    def __init__(self, storage_path: str, material_nodes_path='data/nodes/materials'):
        super().__init__(storage_path=storage_path, initialize_kwargs={'material_nodes_path':material_nodes_path})
        
    def initialize(self, material_nodes_path='data/nodes/materials'):
        """
        Creates Site nodes if no file exists, otherwise loads them from a file.
        """
        self.name_column = 'material_id'
        # Retrieve material nodes with lattice properties
        try:
            material_nodes = MaterialNodes(material_nodes_path)
            
            lattice_names=['structure.lattice.a', 'structure.lattice.b', 'structure.lattice.c', 
                        'structure.lattice.alpha', 'structure.lattice.beta', 'structure.lattice.gamma',
                        'structure.lattice.volume']
            id_names=['id', 'core.material_id']
            tmp_dict={field:[] for field in id_names}
            tmp_dict.update({field:[] for field in lattice_names})
            table=material_nodes.read(columns=['structure.sites', *id_names, *lattice_names])
            # table=material_nodes.read(columns=['structure.sites', *id_names])#, *lattice_names])
            material_sites=table['structure.sites'].combine_chunks()
            

            flatten_material_sites=pc.list_flatten(material_sites)
            material_sites_length_list=pc.list_value_length(material_sites).to_numpy()

            for i, legnth in enumerate(material_sites_length_list):
                for field_name in tmp_dict.keys():
                    column=table[field_name].combine_chunks()
                    value=column[i]
                    tmp_dict[field_name].extend([value]*legnth)
            table=None
            
            
            arrays=flatten_material_sites.flatten()
            names=flatten_material_sites.type.names
            
            flatten_material_sites=None
            material_sites_length_list=None

            for name, column_values in tmp_dict.items():
                arrays.append(pa.array(column_values))
                names.append(name)
                
            table=pa.Table.from_arrays(arrays, names=names)
            
            for i,column in enumerate(table.columns):
                field = table.schema.field(i)
                field_name = field.name
                if '.' in field_name:
                    field_name = field_name.split('.')[-1]
                if 'id' == field_name:
                    field_name = 'material_node_id'
                new_field = field.with_name(field_name)
                table= table.set_column(i,new_field,column)
                
        except Exception as e:
            logger.error(f"Error creating site nodes: {e}")
            return None
        return table
