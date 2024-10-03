import os
import json
from glob import glob
from multiprocessing import Pool
import itertools

import numpy as np
import pandas as pd
import pymatgen.core as pmat
from matminer.datasets import load_dataset
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.structure import XRDPowderPattern
from matminer.featurizers.composition import ElementFraction

from matgraphdb.utils import MP_DIR, DB_DIR, SIMILARITY_DIR,N_CORES
from matgraphdb.utils.math_utils import cosine_similarity
from matgraphdb.utils.general_utils import chunk_list


CHUNK_DIR=os.path.join(SIMILARITY_DIR,'chunks')

def similarity_calc(material_combs_chunk):
    """
    Calculate the similarity between pairs of materials.

    Parameters:
    - material_combs_chunk (tuple): A tuple containing the chunk index and a list of material combinations.

    Returns:
    - None

    This function takes a chunk of material combinations and calculates the similarity between each pair of materials.
    The similarity is calculated based on the structure features of the materials using cosine similarity.

    The calculated similarity values are stored in a dictionary and saved to a JSON file.

    Note: This function assumes that the necessary directories and files are already set up.

    Example usage:
    similarity_calc((0, [('mat1', 'mat2'), ('mat3', 'mat4')]))
    """
    
    # Function code goes here
    # ...
def similarity_calc(material_combs_chunk):

    i_chunk,material_combs=material_combs_chunk
    
    chunk_file=os.path.join(CHUNK_DIR,f'chunk_{i_chunk}.json')

    material_ids=[]
    for material_comb in material_combs:
        mat_1,mat_2=material_comb
        if mat_1 not in material_ids:
            material_ids.append(mat_1)
        if mat_2 not in material_ids:
            material_ids.append(mat_2)

    structures=[]
    compositions=[]
    for material_id in material_ids:
        material_json=os.path.join(DB_DIR,material_id + '.json')

        with open(material_json) as f:
            db = json.load(f)
            struct = pmat.Structure.from_dict(db['structure'])
            structures.append(struct)
            compositions.append(struct.composition)

    structure_data = pd.DataFrame({'structure': structures}, index=material_ids)
    # composition_data = pd.DataFrame({'composition': compositions}, index=material_ids)
    
    structure_featurizer = MultipleFeaturizer([XRDPowderPattern()])
    # composition_featurizer = MultipleFeaturizer([ElementFraction()])

    structure_features = structure_featurizer.featurize_dataframe(structure_data,"structure")
    # composition_features = composition_featurizer.featurize_dataframe(composition_data,"composition")

    structure_features=structure_features.drop(columns=['structure'])
    # composition_features=composition_features.drop(columns=['composition'])


    features=structure_features
    similarity_dict={}
    for material_comb in material_combs:
        mat_1,mat_2=material_comb
        row_1 = features.loc[mat_1].values
        row_2 = features.loc[mat_2].values

        similarity=cosine_similarity(a=row_1,b=row_2)

        pair_name=f'{mat_1}_{mat_2}'
        similarity_dict.update({pair_name : similarity})

    with open(chunk_file,'w') as f:
        json.dump(similarity_dict, f, indent=4)


if __name__=='__main__':
    print('Running Similarity analysis')
    print('Database Dir : ', DB_DIR)
    
    
    CHUNK_SIZE=1000

    database_files=glob(DB_DIR + os.sep +'*.json')
    mpids=[file.split(os.sep)[-1].split('.')[0] for file in database_files]
    # print(mpids)
    material_combs=list(itertools.combinations_with_replacement(mpids[:100], r=2 ))

    print(len(material_combs))
    # print(material_combs[:100])
    material_combs_chunks = chunk_list(material_combs, CHUNK_SIZE)
    material_combs_chunks= [(i,material_combs_chunk) for i,material_combs_chunk in enumerate(material_combs_chunks)]

    print(len(material_combs_chunks))
    # with Pool(N_CORES) as p:
    #     p.map(similarity_analysis, material_combs_chunks)
    for material_combs_chunk in material_combs_chunks:
        similarity_calc(material_combs_chunk)




    # Create empty similarity file
    similarity_file= os.path.join(SIMILARITY_DIR, 'similarity.json')
    tmp_dict={}
    with open(similarity_file,'w') as f:
        try:
            data=json.load(f)
        except:
            json.dump(tmp_dict, f, indent=4)

    chunk_files=glob(CHUNK_DIR + os.sep + '*.json')
    similarity_dict={}
    for chunk_file in chunk_files:
        with open(chunk_file) as f:
            chunk_dict = json.load(f)
        similarity_dict.update(chunk_dict)


    with open(similarity_file,'w') as f:
        json.dump(similarity_dict, f)

        
