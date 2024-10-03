import os
import logging
import numpy as np
import glob
import json

from matgraphdb import config
from matgraphdb.utils.file_utils import load_json, save_parquet, load_parquet

logger = logging.getLogger(__name__)

def get_coord_geom_info():
    coord_geom_dir=os.path.join(config.pkg_dir,'resources','coordination_geometries')
    files=glob.glob(coord_geom_dir + '/*.json')

    cg_list=[]
    mp_symbols={}
    cg_points={}
    mp_coord_encoding={}

    for file in files:

        with open(file) as f:
            dd = json.load(f)
        cg_list.append(dd)
        mp_symbols.update({dd['mp_symbol']:0})
        cg_points.update({dd['mp_symbol']:dd['points']})


        coord_encoding=np.zeros(shape=14)
        coord_num=int(dd['mp_symbol'].split(':')[-1])
        
        if coord_num<=13:
            coord_encoding[coord_num-1]=1
        elif coord_num==20:
            coord_encoding[-1]=1
        mp_coord_encoding.update({dd['mp_symbol']:coord_encoding})

    coord_nums=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,20])
    return mp_coord_encoding, coord_nums

def extract_coordination_encoding(data:dict):
    mp_symbol=data.get('mp_symbol',None)
    if mp_symbol is None:
        logger.info(f"No 'mp_symbol' key found in data: {data}")
        return None
    
    coord_encoding=np.zeros(shape=14)
    coord_num=int(mp_symbol.split(':')[-1])
    if coord_num<=13:
        coord_encoding[coord_num-1]=1
    elif coord_num==20:
        coord_encoding[-1]=1
    return dict(coordination_encoding=coord_encoding, coordination_number=coord_num)


def convert_coord_geo_json_to_parquet(json_files, parquet_file, **kwargs):
    """
    Converts a JSON file containing coordination geometry information to a Parquet file.

    This function reads a JSON file containing coordination geometry information and 
    converts it to a Parquet file. The JSON file should contain a dictionary with 
    keys 'mp_symbol' and 'points', where 'mp_symbol' is the atomic symbol and 'points' 
    is a list of coordinates. The Parquet file will be saved at the specified location.

    Parameters
    ----------
    json_file : str
        The path to the JSON file containing coordination geometry information.
    parquet_file : str
        The path to save the Parquet file.
    kwargs
        Additional keyword arguments to pass to the `save_parquet` function.

    Returns
    -------
    None
    """
    data_list=[]
    for file in json_files:
        data=load_json(file)

        data.update( extract_coordination_encoding(data))
        data_list.append(data)

    table=save_parquet(data_list, parquet_file, **kwargs)
    return table

def load_coord_geometry(parquet_file, **kwargs):
    """
    Load coordination geometry information from a Parquet file.

    This function reads a Parquet file containing coordination geometry information and returns a dictionary
    with atomic symbols as keys and lists of coordinates as values.

    Parameters
    ----------
    parquet_file : str
        The path to the Parquet file containing coordination geometry information.
    kwargs
        Additional keyword arguments to pass to the `load_parquet` function.

    Returns
    -------
    dict
        A dictionary with atomic symbols as keys and lists of coordinates as values.
    """
    table=load_parquet(parquet_file, **kwargs)
    return table




# if __name__ == '__main__':
#     import pandas as pd
#     resources_dir=os.path.join(config.pkg_dir,'utils','chem_utils','resources')
#     # coord_geom_dir=os.path.join(resources_dir,'coordination_geometries')
#     # parquet_file=os.path.join(resources_dir,'coordination_geometries.parquet')
#     # # json_files=glob.glob(coord_geom_dir + '/*.json')
#     # # convert_coord_geo_json_to_parquet(json_files, parquet_file)


#     # table=load_coord_geometry(parquet_file)
#     # df=table.to_pandas()

#     # print(df.head())
#     # df.to_csv(os.path.join(config.data_dir,'coordination_geometries.csv'))


#     df = pd.read_csv(os.path.join(resources_dir,'imputed_periodic_table_values.csv'))
#     save_parquet(df, os.path.join(resources_dir,'imputed_periodic_table_values.parquet'))

#     df = pd.read_csv(os.path.join(resources_dir,'interim_periodic_table_values.csv'))
#     save_parquet(df, os.path.join(resources_dir,'interim_periodic_table_values.parquet'))

#     df = pd.read_csv(os.path.join(resources_dir,'raw_periodic_table_values.csv'))
#     save_parquet(df, os.path.join(resources_dir,'raw_periodic_table_values.parquet'))


