import os
import json

import pandas as pd
import pymatgen.core as pmat

from matgraphdb.database.neo4j.node_types import (ELEMENTS, MAGNETIC_STATES, CRYSTAL_SYSTEMS, CHEMENV_NAMES,
                                                  MATERIAL_FILES, CHEMENV_ELEMENT_NAMES, SPG_NAMES)
from matgraphdb.database.json.utils import PROPERTY_NAMES
from matgraphdb.utils import  GLOBAL_PROP_FILE, NODE_DIR, LOGGER




def create_element_nodes(node_names=ELEMENTS,filepath=None):
    node_dict={
            'elementId:ID(element-ID)':[],
            'type:LABEL':[],
            'name':[],
            'atomic_number':[],
            'atomic_mass':[],
            'X':[],
            'atomic_radius':[],
            'group':[],
            'row':[]
            
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
        node_dict['name'].append(element)
        node_dict['atomic_number'].append(atomic_number)
        node_dict['atomic_mass'].append(atomic_mass)
        node_dict['X'].append(x)
        node_dict['atomic_radius'].append(atomic_radius)
        node_dict['group'].append(group)
        node_dict['row'].append(row)

    df=pd.DataFrame(node_dict)
    if filepath is not None:
        df=pd.DataFrame(node_dict)
        df.to_csv(filepath,index=False)
    return df

def create_crystal_system_nodes(node_names=CRYSTAL_SYSTEMS,filepath=None):
    node_dict={
            'crystalSystemId:ID(crystalSystem-ID)':[],
            'type:LABEL':[],
            'name':[],
            }
    for i,node_name in enumerate(node_names):
        node_dict['crystalSystemId:ID(crystalSystem-ID)'].append(i)
        node_dict['type:LABEL'].append("crystalSystem")
        node_dict['name'].append(node_name)

    df=pd.DataFrame(node_dict)
    if filepath is not None:
        df=pd.DataFrame(node_dict)
        df.to_csv(filepath,index=False)
    return df

def create_chemenv_nodes(node_names=CHEMENV_NAMES,filepath=None):
    node_dict={
            'chemenvId:ID(chemenv-ID)':[],
            'type:LABEL':[],
            'name':[],
            }
    for i,node_name in enumerate(node_names):
        node_name=node_name.replace(':','_')
        node_dict['chemenvId:ID(chemenv-ID)'].append(i)
        node_dict['type:LABEL'].append("chemenv")
        node_dict['name'].append(node_name)

    df=pd.DataFrame(node_dict)
    if filepath is not None:
        df=pd.DataFrame(node_dict)
        df.to_csv(filepath,index=False)
    return df

def create_chemenvElement_nodes(node_names=CHEMENV_ELEMENT_NAMES,filepath=None):
    node_dict={
            'chemenvElementId:ID(chemenvElement-ID)':[],
            'type:LABEL':[],
            'name':[],
            }
    for i,node_name in enumerate(node_names):
        node_name=node_name.replace(':','_')
        node_dict['chemenvElementId:ID(chemenvElement-ID)'].append(i)
        node_dict['type:LABEL'].append("chmenvElement")
        node_dict['name'].append(node_name)

    df=pd.DataFrame(node_dict)
    if filepath is not None:
        df=pd.DataFrame(node_dict)
        df.to_csv(filepath,index=False)
    return df

def create_mag_state_nodes(node_names=MAGNETIC_STATES,filepath=None):
    node_dict={
            'magStateId:ID(magStateId-ID)':[],
            'type:LABEL':[],
            'name':[],
            
            }
    for i,node_name in enumerate(node_names):
        node_dict['magStateId:ID(magStateId-ID)'].append(i)
        node_dict['type:LABEL'].append('magState')
        node_dict['name'].append(node_name)

    df=pd.DataFrame(node_dict)
    if filepath is not None:
        df=pd.DataFrame(node_dict)
        df.to_csv(filepath,index=False)
    return df

def create_materials_nodes(node_names=MATERIAL_FILES,filepath=None):
    node_dict={
            'materialsId:ID(materialsId-ID)':[],
            'type:LABEL':[],
            'name':[],
            }
    node_dict.update({key:[] for key in PROPERTY_NAMES})
    for i,mat_file in enumerate(node_names):
        if i%100==0:
            print(i)

        with open(mat_file) as f:
            db = json.load(f)

        mpid_name=mat_file.split(os.sep)[-1].split('.')[0]
        mpid_name=mpid_name.replace('-','_')

        node_dict['materialsId:ID(materialsId-ID)'].append(i)
        node_dict['type:LABEL'].append("Material")
        node_dict['name'].append(mpid_name)
        for property_name in PROPERTY_NAMES:
            try:
                value=db[property_name]
            except:
                value=None
            node_dict[property_name].append(value)


        

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
    df=create_element_nodes(node_names=ELEMENTS,filepath=os.path.join(save_path,'elements.csv'))
    df=create_crystal_system_nodes(node_names=CRYSTAL_SYSTEMS,filepath=os.path.join(save_path,'crystal_systems.csv'))
    df=create_chemenv_nodes(node_names=CHEMENV_NAMES,filepath=os.path.join(save_path,'chemenv_names.csv'))
    df=create_chemenvElement_nodes(node_names=CHEMENV_ELEMENT_NAMES,filepath=os.path.join(save_path,'chemenv_element_names.csv'))
    df=create_mag_state_nodes(node_names=MAGNETIC_STATES,filepath=os.path.join(save_path,'magnetic_states.csv'))
    df=create_materials_nodes(node_names=MATERIAL_FILES,filepath=os.path.join(save_path,'materials.csv'))
    print('Finished creating nodes')


if __name__ == '__main__':
    main()