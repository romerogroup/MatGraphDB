


from glob import glob
import os
import shutil
import json

from mp_api.client import MPRester

from matgraphdb.utils import DATA_DIR, MP_API_KEY, EXTERNAL_DATA_DIR


def chunk_list(input_list, chunk_size):
    # Create an empty list to hold the chunks
    chunks = []
    # Iterate over the start index of each chunk
    for i in range(0, len(input_list), chunk_size):
        # Append a chunk of the specified size to the chunks list
        chunks.append(input_list[i:i + chunk_size])
    return chunks


def download_materials(save_dir, fileds_to_include):
    """
    Downloads materials from the Materials Project database.
    https://github.com/materialsproject/api/tree/main/mp_api/client/routes/materials

    Args:
        save_dir (str): The directory where the downloaded materials will be saved.
        fileds_to_include (list): A list of fields to include in the downloaded materials.

    Returns:
        None
    """

    with MPRester(MP_API_KEY) as mpr:
        summary_docs = mpr.summary._search( 
                                        nsites_max=40,
                                        energy_above_hull_min=0,
                                        energy_above_hull_max=0.2,
                                        fields=fileds_to_include)

    print('-'*200)
    print("Generating directory json files database")
    print('-'*200)


    os.makedirs(save_dir,exist_ok=True)

    for doc in summary_docs:
        summary_doc_dict = doc.dict()
        mp_id=summary_doc_dict['material_id']

        json_file=os.path.join(save_dir,f'{mp_id}.json')

        json_database_entry={}
        for field_name in fileds_to_include:
            if field_name in summary_doc_dict.keys():
                json_database_entry.update({field_name:summary_doc_dict[field_name]})

        with open(json_file, 'w') as f:
            json.dump(json_database_entry, f, indent=4)

    print('-'*200)


def download_bonds_endpoint(save_dir, fileds_to_include, material_ids):
    """
    Downloads materials from the Materials Project database.
    https://github.com/materialsproject/api/tree/main/mp_api/client/routes/materials
    https://api.materialsproject.org/redoc#section/Accessing-Data

    Args:
        save_dir (str): The directory where the downloaded materials will be saved.
        material_ids (list): A list of fields to include in the downloaded materials.

    Returns:
        None
    """
    chunks = chunk_list(material_ids, chunk_size=10000)
    for chunk in chunks:
        with MPRester(MP_API_KEY) as mpr:
            docs = mpr.materials.bonds.search(material_ids=chunk, all_fields=True)

        print('-'*200)
        print("Generating directory json files database")
        print('-'*200)


        os.makedirs(save_dir,exist_ok=True)

        for doc in docs:
            doc_dict = doc.dict()
            mp_id=doc_dict['material_id']

            json_file=os.path.join(save_dir,f'{mp_id}.json')

            json_database_entry={}
            for field_name in fileds_to_include:
                if field_name in doc_dict.keys():
                    json_database_entry.update({field_name:doc_dict[field_name]})

            with open(json_file, 'w') as f:
                json.dump(json_database_entry, f, indent=4)

        print('-'*200)



def download_elasticity_endpoint(save_dir, fileds_to_include, material_ids):
    """
    Downloads materials from the Materials Project database.
    https://github.com/materialsproject/api/tree/main/mp_api/client/routes/materials
    https://api.materialsproject.org/redoc#section/Accessing-Data

    Args:
        save_dir (str): The directory where the downloaded materials will be saved.
        material_ids (list): A list of fields to include in the downloaded materials.

    Returns:
        None
    """
    chunks = chunk_list(material_ids, chunk_size=10000)
    for chunk in chunks:
        with MPRester(MP_API_KEY) as mpr:
            # docs = mpr.materials.elasticity.search(material_ids=chunk, all_fields=True)
            docs = mpr.materials.elasticity.search(material_ids=chunk, fields=['elastic_tensor'])

        print('-'*200)
        print("Generating directory json files database")
        print('-'*200)


        os.makedirs(save_dir,exist_ok=True)

        for doc in docs:
            doc_dict = doc.dict()
            mp_id=doc_dict['material_id']

            json_file=os.path.join(save_dir,f'{mp_id}.json')

            json_database_entry={}
            for field_name in fileds_to_include:
                if field_name in doc_dict.keys():
                    json_database_entry.update({field_name:doc_dict[field_name]})

            with open(json_file, 'w') as f:
                json.dump(json_database_entry, f, indent=4)

        print('-'*200)


def download_piezoelectric_endpoint(save_dir, fileds_to_include, material_ids):
    """
    Downloads materials from the Materials Project database.
    https://github.com/materialsproject/api/tree/main/mp_api/client/routes/materials
    https://api.materialsproject.org/redoc#section/Accessing-Data

    Args:
        save_dir (str): The directory where the downloaded materials will be saved.
        material_ids (list): A list of fields to include in the downloaded materials.

    Returns:
        None
    """
    chunks = chunk_list(material_ids, chunk_size=10000)
    for chunk in chunks:
        with MPRester(MP_API_KEY) as mpr:
            docs = mpr.materials.piezoelectric.search(material_ids=chunk,all_fields=True)
            # docs = mpr.materials.piezoelectric.search(material_ids=chunk,fields=['ionic','e_ij_max'])


        print('-'*200)
        print("Generating directory json files database")
        print('-'*200)


        os.makedirs(save_dir,exist_ok=True)

        for doc in docs:
            doc_dict = doc.dict()
            mp_id=doc_dict['material_id']

            json_file=os.path.join(save_dir,f'{mp_id}.json')

            json_database_entry={}
            for field_name in fileds_to_include:
                if field_name in doc_dict.keys():
                    json_database_entry.update({field_name:doc_dict[field_name]})

            with open(json_file, 'w') as f:
                json.dump(json_database_entry, f, indent=4)

        print('-'*200)



if __name__=='__main__':



    fileds_to_include = [
        "nsites",
        "elements",
        "nelements",
        "composition",
        "composition_reduced",
        "formula_pretty",
        "formula_anonymous",
        "chemsys",
        "volume",
        "density",
        "density_atomic",
        "symmetry",
        "property_name",
        "material_id",
        "last_updated",
        "origins",
        "warnings",
        "structure",
        "uncorrected_energy_per_atom",
        "energy_per_atom",
        "formation_energy_per_atom",
        "energy_above_hull",
        "is_stable",
        "equilibrium_reaction_energy_per_atom",
        "decomposes_to",
        "xas",
        "grain_boundaries",
        "band_gap",
        "cbm",
        "vbm",
        "efermi",
        "is_gap_direct",
        "is_metal",
        "es_source_calc_id",
        "bandstructure",
        "dos",
        "dos_energy_up",
        "dos_energy_down",
        "is_magnetic",
        "ordering",
        "total_magnetization",
        "total_magnetization_normalized_vol",
        "total_magnetization_normalized_formula_units",
        "num_magnetic_sites",
        "num_unique_magnetic_sites",
        "types_of_magnetic_species",
        "bulk_modulus",
        "shear_modulus",
        "universal_anisotropy",
        "homogeneous_poisson",
        "e_total",
        "e_ionic",
        "e_electronic",
        "n",
        "e_ij_max",
        "weighted_surface_energy_EV_PER_ANG2",
        "weighted_surface_energy",
        "weighted_work_function",
        "surface_anisotropy",
        "shape_factor",
        "has_reconstructed",
        "possible_species",
        "has_props",
        "theoretical"
    ]

    # save_dir=os.path.join(EXTERNAL_DATA_DIR,'materials_project','json_database')
    # download_materials(save_dir=save_dir,fileds_to_include=fileds_to_include)


    fileds_to_include = [
                    "material_id",
                    "last_updated",
                    "origins",
                    "warnings",
                    "structure_graph",
                    "method",
                    "bond_types",
                    "bond_length_stats",
                    "coordination_envs",
                    "coordination_envs_anonymous"]
    external_dir=os.path.join(EXTERNAL_DATA_DIR,'materials_project','json_database')
    material_ids = [file.split('.')[0] for file in os.listdir(external_dir) if file.endswith('.json')]
    # save_dir=os.path.join(EXTERNAL_DATA_DIR,'materials_project','bonds_database')
    # download_bonds_endpoint(save_dir, fileds_to_include, material_ids)


    save_dir=os.path.join(EXTERNAL_DATA_DIR,'materials_project','elasticity_database_2')
    material_ids = ['mp-66']
    download_elasticity_endpoint(save_dir, fileds_to_include,material_ids)


    # save_dir=os.path.join(EXTERNAL_DATA_DIR,'materials_project','piezoelectric_database_2')
    # material_ids = ['mp-648932','mp-4829']
    # download_piezoelectric_endpoint(save_dir, fileds_to_include, material_ids)

