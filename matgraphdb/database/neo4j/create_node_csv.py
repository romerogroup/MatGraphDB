import os
import json
from glob import glob

import pandas as pd
import pymatgen.core as pmat

from matgraphdb.database.neo4j.node_types import (ELEMENTS, MAGNETIC_STATES, CRYSTAL_SYSTEMS, CHEMENV_NAMES,
                                                  MATERIAL_FILES, CHEMENV_ELEMENT_NAMES, SPG_NAMES,OXIDATION_STATES)
from matgraphdb.database.json.utils import PROPERTY_NAMES
from matgraphdb.utils import  GLOBAL_PROP_FILE, NODE_DIR, LOGGER, ENCODING_DIR




def create_element_nodes(node_names=ELEMENTS,filepath=None):
    node_dict={
            'elementId:ID(element-ID)':[],
            'type:LABEL':[],
            'name:string':[],
            'atomic_number:int':[],
            'atomic_mass:float':[],
            'X:float':[],
            'atomic_radius:float':[],
            'group:int':[],
            'row:int':[]
            
            }
    for i,element in enumerate(node_names[:]):
        # pymatgen object. Given element string, will have useful properties 
        pmat_element = pmat.periodic_table.Element(element)

        # Handling None and nan value cases
        if str(pmat_element.Z) != 'nan':
            atomic_number=pmat_element.Z
        else:
            atomic_number=None
        if str(pmat_element.X) != 'nan':
            x=pmat_element.X
        else:
            x=None
        if str(pmat_element.atomic_radius) != 'nan' and str(pmat_element.atomic_radius) != 'None':
            atomic_radius=float(pmat_element.atomic_radius)
        else:
            atomic_radius=None
        if str(pmat_element.group) != 'nan':
            group=pmat_element.group
        else:
            group=None
        if str(pmat_element.row) != 'nan':
            row=pmat_element.row
        else:
            row=None
        if str(pmat_element.atomic_mass) != 'nan':
            atomic_mass=float(pmat_element.atomic_mass)
        else:
            atomic_mass=None

        node_dict['elementId:ID(element-ID)'].append(i)
        node_dict['type:LABEL'].append("Element")
        node_dict['name:string'].append(element)
        node_dict['atomic_number:int'].append(atomic_number)
        node_dict['atomic_mass:float'].append(atomic_mass)
        node_dict['X:float'].append(x)
        node_dict['atomic_radius:float'].append(atomic_radius)
        node_dict['group:int'].append(group)
        node_dict['row:int'].append(row)

    df=pd.DataFrame(node_dict)
    if filepath is not None:
        df=pd.DataFrame(node_dict)
        df.to_csv(filepath,index=False)
    return df

def create_crystal_system_nodes(node_names=CRYSTAL_SYSTEMS,filepath=None):
    node_dict={
            'crystalSystemId:ID(crystalSystem-ID)':[],
            'type:LABEL':[],
            'name:string':[],
            }
    for i,node_name in enumerate(node_names):
        node_dict['crystalSystemId:ID(crystalSystem-ID)'].append(i)
        node_dict['type:LABEL'].append("crystalSystem")
        node_dict['name:string'].append(node_name)

    df=pd.DataFrame(node_dict)
    if filepath is not None:
        df=pd.DataFrame(node_dict)
        df.to_csv(filepath,index=False)
    return df

def create_chemenv_nodes(node_names=CHEMENV_NAMES,filepath=None):
    node_dict={
            'chemenvId:ID(chemenv-ID)':[],
            'type:LABEL':[],
            'name:string':[],
            }
    for i,node_name in enumerate(node_names):
        node_name=node_name.replace(':','_')
        node_dict['chemenvId:ID(chemenv-ID)'].append(i)
        node_dict['type:LABEL'].append("chemenv")
        node_dict['name:string'].append(node_name)

    df=pd.DataFrame(node_dict)
    if filepath is not None:
        df=pd.DataFrame(node_dict)
        df.to_csv(filepath,index=False)
    return df

def create_chemenvElement_nodes(node_names=CHEMENV_ELEMENT_NAMES,filepath=None):
    node_dict={
            'chemenvElementId:ID(chemenvElement-ID)':[],
            'type:LABEL':[],
            'name:string':[],
            }
    for i,node_name in enumerate(node_names):
        node_name=node_name.replace(':','_')
        node_dict['chemenvElementId:ID(chemenvElement-ID)'].append(i)
        node_dict['type:LABEL'].append("chmenvElement")
        node_dict['name:string'].append(node_name)

    df=pd.DataFrame(node_dict)
    if filepath is not None:
        df=pd.DataFrame(node_dict)
        df.to_csv(filepath,index=False)
    return df

def create_mag_state_nodes(node_names=MAGNETIC_STATES,filepath=None):
    node_dict={
            'magStateId:ID(magStateId-ID)':[],
            'type:LABEL':[],
            'name:string':[],
            
            }
    for i,node_name in enumerate(node_names):
        node_dict['magStateId:ID(magStateId-ID)'].append(i)
        node_dict['type:LABEL'].append('magState')
        node_dict['name:string'].append(node_name)

    df=pd.DataFrame(node_dict)
    if filepath is not None:
        df=pd.DataFrame(node_dict)
        df.to_csv(filepath,index=False)
    return df

def create_spg_nodes(node_names=SPG_NAMES,filepath=None):
    node_dict={
            'spgId:ID(spgId-ID)':[],
            'type:LABEL':[],
            'name:string':[],
            
            }
    for i,node_name in enumerate(node_names):
        node_dict['spgId:ID(spgId-ID)'].append(i)
        node_dict['type:LABEL'].append('spg')
        node_dict['name:string'].append(node_name)

    df=pd.DataFrame(node_dict)
    if filepath is not None:
        df=pd.DataFrame(node_dict)
        df.to_csv(filepath,index=False)
    return df

def create_materials_nodes(node_names=MATERIAL_FILES,filepath=None):
    node_dict={
            'materialsId:ID(materialsId-ID)':[],
            'type:LABEL':[],
            'name:string':[],
            }
    # Add the properties to the node_dict
    for property in PROPERTY_NAMES:
        if property[0]=='symmetry':
            node_dict.update({'crystal_system:string' :[],
                              'space_group:int':[],
                              'point_group:string':[],
                              'symbol:string':[]})
        else:
            node_dict.update({f'{property[0]}:{property[1]}':[]})

    for i,mat_file in enumerate(node_names):
        if i%100==0:
            print(i)

        with open(mat_file) as f:
            db = json.load(f)

        mpid_name=mat_file.split(os.sep)[-1].split('.')[0]
        mpid_name=mpid_name.replace('-','_')

        # Add the properties to the node_dict
        node_dict['materialsId:ID(materialsId-ID)'].append(i)
        node_dict['type:LABEL'].append("Material")
        node_dict['name:string'].append(mpid_name)
        for property in PROPERTY_NAMES:
            # If the property is a symmetry property
            if property[0]=='symmetry':
                try:
                    sym_dict=db[property[0]]

                    node_dict['crystal_system:string'].append(sym_dict['crystal_system'])
                    node_dict['space_group:int'].append(sym_dict['number'])
                    node_dict['point_group:string'].append(sym_dict['point_group'])
                    node_dict['symbol:string'].append(sym_dict['symbol'])
                except:
                    node_dict['crystal_system:string'].append(None)
                    node_dict['space_group:int'].append(None)
                    node_dict['point_group:string'].append(None)
                    node_dict['symbol:string'].append(None)

            # If the property is not a symmetry property
            else:
                try:
                    value=db[property[0]]
                except:
                    value=None
                node_dict[f'{property[0]}:{property[1]}'].append(value)


    # Check if encodings are present
    if os.path.exists(ENCODING_DIR):
        encoding_files=glob(os.path.join(ENCODING_DIR,'*.csv'))
        for encoding_file in encoding_files:
            encoding_name=encoding_file.split(os.sep)[-1].split('.')[0]

            df=pd.read_csv(encoding_file,index_col=0)

            # Convert the dataframe values to a list of strings where the strings are the rows of the dataframe separated by a semicolon
            df = df.apply(lambda x: ';'.join(map(str, x)), axis=1)

            node_dict.update({f'{encoding_name}:float[]': df.tolist()})
        del df

    df=pd.DataFrame(node_dict)

    # Remove rows with nan values. Might need to change this in the futrue
    for encoding_name in encoding_files:
        encoding_name=encoding_name.split(os.sep)[-1].split('.')[0]

        # Where the encoding contains nan value replace with None:
        df[f'{encoding_name}:float[]']=df[f'{encoding_name}:float[]'].apply(lambda x: None if 'nan' in x else x)

        # Remove rows with that contain None values
        df=df.dropna(subset=[f'{encoding_name}:float[]'])

    if filepath is not None:
        df.to_csv(filepath,index=False)
    return df

def create_oxidation_state_nodes(node_names=SPG_NAMES,filepath=None):
    node_dict={
            'oxiStateId:ID(oxiStateId-ID)':[],
            'type:LABEL':[],
            'name:string':[],
            
            }
    for i,node_name in enumerate(node_names):
        node_dict['oxiStateId:ID(oxiStateId-ID)'].append(i)
        node_dict['type:LABEL'].append('Oxidation State')
        node_dict['name:string'].append(node_name)

    df=pd.DataFrame(node_dict)
    if filepath is not None:
        df=pd.DataFrame(node_dict)
        df.to_csv(filepath,index=False)
    return df

def main():
    
    save_path=os.path.join(NODE_DIR)
    print('Save_path : ', save_path)
    os.makedirs(save_path,exist_ok=True)
    print('Creating nodes...')
    # df=create_element_nodes(node_names=ELEMENTS,filepath=os.path.join(save_path,'elements.csv'))
    # df=create_crystal_system_nodes(node_names=CRYSTAL_SYSTEMS,filepath=os.path.join(save_path,'crystal_systems.csv'))
    # df=create_chemenv_nodes(node_names=CHEMENV_NAMES,filepath=os.path.join(save_path,'chemenv_names.csv'))
    # df=create_chemenvElement_nodes(node_names=CHEMENV_ELEMENT_NAMES,filepath=os.path.join(save_path,'chemenv_element_names.csv'))
    # df=create_mag_state_nodes(node_names=MAGNETIC_STATES,filepath=os.path.join(save_path,'magnetic_states.csv'))
    # df=create_spg_nodes(node_names=SPG_NAMES,filepath=os.path.join(save_path,'spg.csv'))
    # df=create_materials_nodes(node_names=MATERIAL_FILES,filepath=os.path.join(save_path,'materials.csv'))
    df=create_oxidation_state_nodes(node_names=OXIDATION_STATES,filepath=os.path.join(save_path,'oxidation_states.csv'))
    print('Finished creating nodes')


if __name__ == '__main__':
    main()