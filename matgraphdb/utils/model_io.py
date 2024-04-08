import os
import json

from pandas import DataFrame
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

from matgraphdb.utils import DB_DIR,MP_DIR,DATASETS_DIR
from matgraphdb.data.manager import DatabaseManager



def cdvae_processing(file):
    with open(file) as f:
        db = json.load(f)

    mpid=db['material_id']
    pretty_formula=db['formula_pretty']
    band_gap=db['band_gap']
    e_above_hull=db['energy_above_hull']
    formation_energy=db['formation_energy_per_atom']
    elements=db['elements']
    spg_number=db['symmetry']['number']

    structure=Structure.from_dict(db['structure'])

    cif_writer=CifWriter(struct=structure)
    cif=cif_writer.__str__()

    return (mpid,formation_energy,band_gap,pretty_formula,e_above_hull,elements,cif,spg_number)

def cdvae_dataset(filepath):

    db=DatabaseManager()
    results=db.process_task(func=cdvae_processing, list=db.database_files)


    df = DataFrame(results, columns=['material_id', 'formation_energy_per_atom', 'band_gap', 'pretty_formula', 'e_above_hull','elements','cif','spacegroup'])

    if filepath:
        df.to_csv(filepath)

    return df

if __name__=='__main__':
    cdvae_dataset(filepath=os.path.join(DATASETS_DIR,'cdvae_matgraphdb.csv'))