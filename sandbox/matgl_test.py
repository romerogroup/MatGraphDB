# import matgl
# print(matgl.get_available_pretrained_models())
import os

from pymatgen.core import Lattice, Structure
import matgl
import torch
import numpy as np
import pandas as pd
print(torch.cuda.is_available())
print(matgl.get_available_pretrained_models())
import json

from matgraphdb.utils import ENCODING_DIR, DB_DIR
from pymatgen.core import Structure

# model = matgl.load_model("MEGNet-MP-2018.6.1-Eform")

# # This is the structure obtained from the Materials Project.
# struct = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.1437), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
# vec = model.predict_structure(struct)
# print(vec.shape)
# print(f"The predicted formation energy for CsCl is {float(eform.numpy()):.3f} eV/atom.")



def process_megnet(filepath=None, modelname="MEGNet-MP-2018.6.1-Eform"):
    model = matgl.load_model(modelname)
    encodings=np.zeros(shape=(len(os.listdir(DB_DIR)[:]),160 ))
    mpids = []
    for i,material_file in enumerate(os.listdir(DB_DIR)[:]):
        # Do some processing here
        # Load material data from file
        with open(os.path.join(DB_DIR,material_file)) as f:
            db = json.load(f)
            structure=Structure.from_dict(db['structure'])

        mpid=material_file.split(os.sep)[-1].split('.')[0]
        vec = model.predict_structure(structure)

        encodings[i,:]=vec

        mpids.append(mpid)

    # Create dataframe with mpids as index column and encodings dim=1 as columns
    df =pd.DataFrame(encodings,
                     columns=[f'{i}' for i in range(160)],
                     index=mpids)

    if filepath is not None:
        df.to_csv(filepath, index=True)
    return df

if __name__ == '__main__':
    modelname="MEGNet-MP-2018.6.1-Eform"
    df=process_megnet(filepath=os.path.join(ENCODING_DIR,f'{modelname}.csv'), modelname=modelname)

    # modelname="M3GNet-MP-2021.2.8-DIRECT-PES"
    # df=process_megnet(filepath=os.path.join(ENCODING_DIR,f'{modelname}.csv'), modelname=modelname)

    # modelname="M3GNet-MP-2021.2.8-PES"
    # df=process_megnet(filepath=os.path.join(ENCODING_DIR,f'{modelname}.csv'), modelname=modelname)

    # modelname="M3GNet-MP-2018.6.1-Eform"
    # df=process_megnet(filepath=os.path.join(ENCODING_DIR,f'{modelname}.csv'), modelname=modelname)
    
    # modelname="MEGNet-MP-2019.4.1-BandGap-mfi"
    # df=process_megnet(filepath=os.path.join(ENCODING_DIR,f'{modelname}.csv'), modelname=modelname)