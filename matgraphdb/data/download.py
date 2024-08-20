from glob import glob
import os
import shutil
import json
from dotenv import load_dotenv

from mp_api.client import MPRester

from matgraphdb.utils import DATA_DIR, EXTERNAL_DATA_DIR

load_dotenv()

MP_API_KEY=os.getenv('MP_API_KEY')

def chunk_list(input_list, chunk_size):
    # Create an empty list to hold the chunks
    chunks = []
    # Iterate over the start index of each chunk
    for i in range(0, len(input_list), chunk_size):
        # Append a chunk of the specified size to the chunks list
        chunks.append(input_list[i:i + chunk_size])
    return chunks


def download_materials(save_dir,**kwargs):
    """
    Downloads materials from the Materials Project database.
    https://github.com/materialsproject/api/tree/main/mp_api/client/routes/materials

    Args:
        save_dir (str): The directory where the downloaded materials will be saved.
        kwargs (dict): A dictionary of keyword arguments to pass to the MPRester.summary._search method.

    Returns:
        None
    """

    with MPRester(MP_API_KEY) as mpr:
        summary_docs = mpr.materials.summary._search(all_fields=True, **kwargs)

    print('-'*200)
    print("Generating directory json files database")
    print('-'*200)


    os.makedirs(save_dir,exist_ok=True)

    for doc in summary_docs:
        # summary_doc_dict = doc.model_dump(mode='json')
        summary_doc_dict = doc.dict()
        mp_id=summary_doc_dict['material_id']

        json_file=os.path.join(save_dir,f'{mp_id}.json')

        with open(json_file, 'w') as f:
            json.dump(summary_doc_dict, f, indent=4)

    print('-'*200)

def download_materials_data(save_dir,  endpoint, material_ids):
    chunks = chunk_list(material_ids, chunk_size=10000)

    final_save_dir=os.path.join(save_dir,endpoint)
    os.makedirs(final_save_dir,exist_ok=True)

    for chunk in chunks:
        with MPRester(MP_API_KEY) as mpr:
            docs = eval(f"mpr.materials.{endpoint}.search(material_ids=chunk, all_fields=True)")

        print('-'*200)
        print("Generating directory json files database")
        print('-'*200)


        for doc in docs:
            # doc_dict = doc.model_dump(mode='json')
            doc_dict = doc.dict()
            mp_id=doc_dict['material_id']

            json_file=os.path.join(final_save_dir,f'{mp_id}.json')


            with open(json_file, 'w') as f:
                json.dump(doc_dict, f, indent=4)

        print('-'*200)

def download_molecules(save_dir,**kwargs):
    """
    Downloads materials from the Materials Project database.
    https://github.com/materialsproject/api/tree/main/mp_api/client/routes/materials

    Args:
        save_dir (str): The directory where the downloaded materials will be saved.
        kwargs (dict): A dictionary of keyword arguments to pass to the MPRester.summary._search method.

    Returns:
        None
    """

    with MPRester(MP_API_KEY) as mpr:
        summary_docs = mpr.molecules.summary.search(fields=['molecule_id'])

    molecules_ids = [doc.molecule_id for doc in summary_docs]
    with MPRester(MP_API_KEY) as mpr:
        summary_docs = mpr.molecules.summary.search(molecules_ids=molecules_ids, all_fields=True, **kwargs)
    
    print('-'*200)
    print("Generating directory json files database")
    print('-'*200)


    os.makedirs(save_dir,exist_ok=True)

    for doc in summary_docs:
        summary_doc_dict = doc.model_dump(mode='json')
        mp_id=summary_doc_dict['material_id']

        json_file=os.path.join(save_dir,f'{mp_id}.json')

        with open(json_file, 'w') as f:
            json.dump(summary_doc_dict, f, indent=4)

    print('-'*200)


def download_molecules_data(save_dir, endpoint, molecules_ids):
    chunks = chunk_list(molecules_ids, chunk_size=10000)

    final_save_dir=os.path.join(save_dir,endpoint)

    os.makedirs(final_save_dir,exist_ok=True)
    for chunk in chunks:
        with MPRester(MP_API_KEY) as mpr:
            docs = eval(f"mpr.molecules.{endpoint}.search(molecules_ids=chunk, all_fields=True)")
        print('-'*200)
        print("Generating directory json files database")
        print('-'*200)

        for doc in docs:
            doc_dict = doc.model_dump(mode='json')
            mp_id=doc_dict['molecules_ids']

            json_file=os.path.join(final_save_dir,f'{mp_id}.json')

            with open(json_file, 'w') as f:
                json.dump(doc_dict, f, indent=4)

        print('-'*200)

if __name__=='__main__':


    # Using the Mprester API
    # with MPRester(api_key=MP_API_KEY) as mpr:
    #     elasticity_doc = mpr.elasticity.search(material_ids=["mp-66"])
    #     print(dir(elasticity_doc[0]))
    #     print(elasticity_doc)
    ################################################################################################
    ################################################################################################
    # Download materials
    ################################################################################################
    ################################################################################################

    # save_dir=os.path.join(EXTERNAL_DATA_DIR,'materials_project','json_database')

    # materials_filter={
    #                 'nsites_max':40,
    #                 'energy_above_hull_min':0,
    #                 'energy_above_hull_max':0.2
    #                 }
    # download_materials(save_dir=save_dir,**materials_filter)


    external_dir=os.path.join(EXTERNAL_DATA_DIR,'materials_project','json_database')
    material_ids = [file.split('.')[0] for file in os.listdir(external_dir) if file.endswith('.json')]

    save_dir=os.path.join(EXTERNAL_DATA_DIR,'materials_project')

    download_materials_data(save_dir, endpoint='bonds', material_ids=material_ids)
    download_materials_data(save_dir, endpoint='elasticity', material_ids=material_ids)
    download_materials_data(save_dir, endpoint='piezoelectric', material_ids=material_ids)
    download_materials_data(save_dir, endpoint='thermo', material_ids=material_ids)
    download_materials_data(save_dir, endpoint='dielectric', material_ids=material_ids)
    download_materials_data(save_dir, endpoint='oxidation_states', material_ids=material_ids)
    download_materials_data(save_dir, endpoint='phonon', material_ids=material_ids)





    ################################################################################################
    ################################################################################################
    # Download molecules
    ################################################################################################
    ################################################################################################

    # download_materials(save_dir=os.path.join(EXTERNAL_DATA_DIR,'molecules','json_database'))
    # external_dir=os.path.join(EXTERNAL_DATA_DIR,'molecules','json_database')
    # molecules_ids = [file.split('.')[0] for file in os.listdir(external_dir) if file.endswith('.json')]

    # save_dir=os.path.join(EXTERNAL_DATA_DIR,'molecules')

    # download_molecules_data(save_dir, endpoint='jcesr', molecules_ids=molecules_ids)
    # download_molecules_data(save_dir, endpoint='redox', molecules_ids=molecules_ids)
    # download_molecules_data(save_dir, endpoint='vibrations', molecules_ids=molecules_ids)
    # download_molecules_data(save_dir, endpoint='thermo', molecules_ids=molecules_ids)
    # download_molecules_data(save_dir, endpoint='bonding', molecules_ids=molecules_ids)
    # download_molecules_data(save_dir, endpoint='orbitals', molecules_ids=molecules_ids)
