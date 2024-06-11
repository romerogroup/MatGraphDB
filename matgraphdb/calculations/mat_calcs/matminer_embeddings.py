import os
import json

import openai
import tiktoken
import pandas as pd
import numpy as np
from matminer.datasets import load_dataset
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.structure import XRDPowderPattern
from matminer.featurizers.composition import ElementFraction
from pymatgen.core import Structure

from matgraphdb.utils import DB_DIR, ENCODING_DIR
from matgraphdb.data import DatabaseManager


def generate_composition_embeddingings():
    compositions=[]
    material_ids=[]
    db=DatabaseManager()
    for material_file in db.database_files[:]:
        material_id=material_file.split(os.sep)[-1].split('.')[0]
        with open(material_file) as f:
            db = json.load(f)
            struct = Structure.from_dict(db['structure'])
            compositions.append(struct.composition)

        material_ids.append(material_id)

    composition_data = pd.DataFrame({'composition': compositions}, index=material_ids)
    composition_featurizer = MultipleFeaturizer([ElementFraction()])
    composition_features = composition_featurizer.featurize_dataframe(composition_data,"composition")
    composition_features=composition_features.drop(columns=['composition'])
    features=composition_features
    for index, row in features.iterrows():
        encoding_file=os.path.join(ENCODING_DIR,index+'.json')

        embedding_dict={'element_fraction':row.values.tolist()}
        if os.path.exists(encoding_file):
            
            with open(encoding_file) as f:
                db = json.load(f)
            db.update(embedding_dict)
            with open(encoding_file,'w') as f:
                json.dump(db, f, indent=None)

        else:
            with open(encoding_file,'w') as f:
                json.dump(embedding_dict, f, indent=None)

if __name__=='__main__':
    generate_composition_embeddingings()