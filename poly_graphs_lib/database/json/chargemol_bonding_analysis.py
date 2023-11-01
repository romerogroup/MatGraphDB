import os
import re
import json
from glob import glob

import pymatgen.core as pmat

from poly_graphs_lib.utils.periodic_table import covalent_cutoff_map
from poly_graphs_lib.database.json import process_database
from poly_graphs_lib.database import DB_DIR,DB_CALC_DIR

BOND_ORDER_CUTOFF=0.20

def bonding_analysis(file, from_scratch=False):

    try:
        with open(file) as f:
            db = json.load(f)
            struct = pmat.Structure.from_dict(db['structure'])
        
        if 'chargemol_bonding_connections' not in db or from_scratch:
            mp_id=file.split(os.sep)[-1].split('.')[0]
            calc_dir=os.path.join(DB_CALC_DIR,mp_id,'static')
            bond_order_file=os.path.join(calc_dir,'DDEC6_even_tempered_bond_orders.xyz')

            with open(bond_order_file,'r') as f:
                text=f.read()


            bond_blocks=re.findall('(?<=Printing BOs for ATOM).*\n([\s\S]*?)(?=The sum of bond orders for this atom is SBO)',text)

            bonding_connections=[]
            for bond_block in bond_blocks:

                bonds=bond_block.strip().split('\n')

                bond_orders=[]
                atom_indices=[]
                for bond in bonds:
                    bond_order=float(re.findall('bond order\s=\s*([.0-9-]*)\s*',bond)[0])

                    # shift so index starts at 0
                    atom_index=int(re.findall('translated image of atom number\s*([0-9]*)\s*',bond)[0]) -1
                    if bond_order >= BOND_ORDER_CUTOFF:
                        bond_orders.append(bond_order)
                        atom_indices.append(atom_index)

                bonding_connections.append(atom_indices)

            
                db['chargemol_bonding_connections']=bonding_connections


    except Exception as e:
        print(e)
        print(file)
        db['chargemol_bonding_connections']=None

    with open(file,'w') as f:
        json.dump(db, f, indent=4)



if __name__=='__main__':
    print('Running Chargemol Bonding analysis')
    print('Database Dir : ', DB_DIR)
    process_database(bonding_analysis)
    # database_files=glob(DB_DIR + os.sep +'*.json')

    # bonding_analysis(database_files[0])
