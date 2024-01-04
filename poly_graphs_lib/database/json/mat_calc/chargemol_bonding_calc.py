import os
import re
import json
from glob import glob



from multiprocessing import Lock


import pymatgen.core as pmat

from poly_graphs_lib.utils.periodic_table import covalent_cutoff_map
from poly_graphs_lib.database.json.utils import process_database
from poly_graphs_lib.database.json import DB_DIR,DB_CALC_DIR
from poly_graphs_lib.utils import LOG_DIR


CHARGEMOL_LOG_FILE = os.path.join(LOG_DIR,'calculations','chargemol','nelements_3','chargemol_bonding_orders_debug.txt')

BOND_ORDER_CUTOFF=0.0


def chargemol_bonding_calc_task(file, from_scratch=True,lock = Lock()):

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
            bonding_orders=[]

            for bond_block in bond_blocks:

                bonds=bond_block.strip().split('\n')

                bond_orders=[]
                atom_indices=[]
                # Catches cases where there are no bonds
                if bonds[0]!='':
                    for bond in bonds:

                        bond_order=float(re.findall('bond order\s=\s*([.0-9-]*)\s*',bond)[0])

                        # shift so index starts at 0
                        atom_index=int(re.findall('translated image of atom number\s*([0-9]*)\s*',bond)[0]) -1
                        if bond_order >= BOND_ORDER_CUTOFF:
                            bond_orders.append(bond_order)
                            atom_indices.append(atom_index)
                else:
                    pass

                bonding_connections.append(atom_indices)
                bonding_orders.append(bond_orders)

            
            db['chargemol_bonding_connections']=bonding_connections
            db['chargemol_bonding_orders']=bonding_orders


    except Exception as e:
        LOGGER.error(f"Error processing file {mpid}: {e}")

        with lock:
            with open(CHARGEMOL_LOG_FILE, 'a') as log_file:
                log_file.write(f"Error in file {file}: {e}\n")

        db['chargemol_bonding_connections']=None
        db['chargemol_bonding_orders']=None

    with open(file,'w') as f:
        json.dump(db, f, indent=4)

def chargemol_bonding_calc():
    LOGGER.info('#'*100)
    LOGGER.info('Running Chargemol Bonding Calculation')
    LOGGER.info('#'*100)

    if os.path.exists(CHARGEMOL_LOG_FILE):
        os.remove(CHARGEMOL_LOG_FILE)
        
    process_database(chargemol_bonding_calc_task)

if __name__=='__main__':
    chargemol_bonding_calc()
