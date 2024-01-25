import os
import json
import time

from matgraphdb.database.neo4j.populate.nodes import Node
from matgraphdb.database.neo4j.node_types import MATERIAL_FILES
from matgraphdb.database.neo4j.utils import execute_statements


PROPERTY_NAMES=[
    "material_id",
    "nsites",
    "elements",
    "nelements",
    "composition",
    "composition_reduced",
    "formula_pretty",
    "volume",
    "density",
    "density_atomic",
    "symmetry",
    "energy_per_atom",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "band_gap",
    "cbm",
    "vbm",
    "efermi",
    "is_gap_direct",
    "is_metal",
    "is_magnetic",
    "ordering",
    "total_magnetization",
    "total_magnetization_normalized_vol",
    "num_magnetic_sites",
    "num_unique_magnetic_sites",
    "k_voigt",
    "k_reuss",
    "k_vrh",
    "g_voigt",
    "g_reuss",
    "g_vrh",
    "universal_anisotropy",
    "homogeneous_poisson",
    "e_total",
    "e_ionic",
    "e_electronic",
    "wyckoffs"
]

def populate_material_nodes(material_files=MATERIAL_FILES):
    print("Populating material nodes...")

    create_statements = []

    unique_constraint="CREATE CONSTRAINT UNIQUE_STRUCTURE_NAME IF NOT EXISTS ON (s:Structure) ASSERT s.name IS UNIQUE;"
    create_statements.append(unique_constraint)
    
    for i,mat_file in enumerate(material_files):
        if i%100==0:
            print(i)

        with open(mat_file) as f:
            db = json.load(f)

        mpid_name=mat_file.split(os.sep)[-1].split('.')[0]
        mpid_name=mpid_name.replace('-','_')
        node=Node(node_name=mpid_name,class_name='Structure')

        for property_name in PROPERTY_NAMES:
            key=property_name
            value=db[key]

            if key == 'structure':
                pass
            elif key=='sites':
                pass   
            elif key=='coordination_connections':
                pass
            elif key=='bonding_cutoff_connections':
                pass
            elif key=='composition_reduced' or key=='composition':
                comp_reduced=[ db[key][element] for element in db['elements']]
                node.add_property(name=key,value=comp_reduced)
            elif key=='symmetry':
                for symmetry_key in db[key].keys():
                    if symmetry_key=='version':
                        pass
                    elif symmetry_key=='crystal_system':
                        node.add_property(name=symmetry_key,value=db[key][symmetry_key].lower())
                    else:
                        node.add_property(name=symmetry_key,value=db[key][symmetry_key])
            elif key=='coordination_environments':
                try:
                    coord_envs=[coord_env[0]['ce_symbol'].replace(':','_') for coord_env in db[key]]
                    node.add_property(name='ce_symbol',value=coord_envs)
                except:
                    pass
            elif key=='coordination_environments_multi_weight':
                try:
                    coord_envs=[coord_env[0]['ce_symbol'].replace(':','_') for coord_env in db[key]]
                    node.add_property(name='ce_symbol',value=coord_envs)
                except:
                    pass
            else:
                node.add_property(name=key,value=value)

        execute_statement=node.final_execute_statement()
        create_statements.append(execute_statement)

    print("Finished populating...")
    return create_statements


def main():
    create_statements=populate_material_nodes(MATERIAL_FILES)
    execute_statements(create_statements)


if __name__ == '__main__':
    main()
