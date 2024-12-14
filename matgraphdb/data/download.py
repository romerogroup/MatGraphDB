from glob import glob
import os
import shutil
import json
from dotenv import load_dotenv
import time
from mp_api.client import MPRester
from parquetdb import ParquetDB, config

config.logging_config.loggers.parquetdb.level='DEBUG'
config.apply()

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


def download_materials(save_dir, chunk_size=10000, **kwargs):
def download_materials(save_dir, chunk_size=10000, **kwargs):
    """
    Downloads materials from the Materials Project database and saves them in chunks of 10000 materials per JSON file.
    https://github.com/materialsproject/api/tree/main/mp_api/client/routes/materials

    Args:
        save_dir (str): The directory where the downloaded materials will be saved.
        kwargs (dict): A dictionary of keyword arguments to pass to the MPRester.summary._search method.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    
    with MPRester(MP_API_KEY) as mpr:
        summary_docs = mpr.materials.summary._search(all_fields=True, **kwargs)

    print("Generating chunked json files database")
    summary_doc_chunks = chunk_list(summary_docs, chunk_size)
    for i_chunk,doc_chunk in enumerate(summary_doc_chunks):
        materials_database_dict={'entries':[]}
        for doc in doc_chunk:
            materials_database_dict['entries'].append(doc.dict())

        json_file = os.path.join(save_dir, f'materials_chunk_{i_chunk}.json')
        with open(json_file, 'w') as f:
            json.dump(materials_database_dict, f)
            
        print(f"Saved {json_file}")

def download_materials_data(save_dir,  endpoint, material_ids, chunk_size=10000, use_search_docs=False):
    """The size is limited by the MPRester API, which is 10000 materials per request.
    """
    chunks = chunk_list(material_ids, chunk_size=chunk_size)

    final_save_dir = os.path.join(save_dir, endpoint)
    os.makedirs(final_save_dir, exist_ok=True)

    for i_chunk,chunk in enumerate(chunks[:]):
        
        with MPRester(MP_API_KEY) as mpr:
            if use_search_docs:
                docs = eval(f"mpr.materials.{endpoint}.search_docs(material_ids=chunk, all_fields=True)")
            else:
                docs = eval(f"mpr.materials.{endpoint}.search(material_ids=chunk, all_fields=True)")
        print("Generating directory json files database")

        materials_database_dict={'entries':[]}
        for doc in docs[:]:
            materials_database_dict['entries'].append(doc.dict())
            
        json_file = os.path.join(final_save_dir, f'{endpoint}_chunk_{i_chunk}.json')
        with open(json_file, 'w') as f:
            json.dump(materials_database_dict, f)

        print('-'*200)

def download_molecules(save_dir, chunk_size=10000):
    """
    Downloads materials from the Materials Project database.
    https://github.com/materialsproject/api/tree/main/mp_api/client/routes/materials

    Args:
        save_dir (str): The directory where the downloaded materials will be saved.
        kwargs (dict): A dictionary of keyword arguments to pass to the MPRester.summary._search method.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    

    with MPRester(MP_API_KEY, monty_decode=False, use_document_model=False) as mpr:
        summary_docs = mpr.molecules.summary.search()
        with open(os.path.join(save_dir, 'molecules_summary.json'), 'w') as f:
            json.dump(summary_docs, f)

    # summary_doc_chunks = chunk_list(summary_docs, chunk_size)

    
    # for i_chunk,doc_chunk in enumerate(summary_doc_chunks):
    #     database_dict={'entries':[]}
    #     for doc in doc_chunk:
    #         database_dict['entries'].append(doc.dict())

    #     json_file = os.path.join(save_dir, f'molecules_chunk_{i_chunk}.json')
    #     with open(json_file, 'w') as f:
    #         json.dump(database_dict, f)
            
    #     print(f"Saved {json_file}")

def download_molecules_data(save_dir, endpoint, molecules_ids, chunk_size=10000):
    chunks = chunk_list(molecules_ids, chunk_size=chunk_size)

    final_save_dir=os.path.join(save_dir,endpoint)
    os.makedirs(final_save_dir,exist_ok=True)
    
    for i_chunk,chunk in enumerate(chunks[:]):
        with MPRester(MP_API_KEY) as mpr:
            docs = eval(f"mpr.molecules.{endpoint}.search(molecules_ids=chunk, all_fields=True)")
        print("Generating directory json files database")

        materials_database_dict={'entries':[]}
        for doc in docs[:]:
            materials_database_dict['entries'].append(doc.dict())
            
        json_file = os.path.join(final_save_dir, f'{endpoint}_chunk_{i_chunk}.json')
        with open(json_file, 'w') as f:
            json.dump(materials_database_dict, f)

        print('-'*200)
        

    

if __name__=='__main__':
    
    
    external_dir=os.path.join('data','external')
    
    materials_dir = os.path.join(external_dir,'materials_project','materials')
    materials_parquetdb_dir=os.path.join(external_dir,'materials_project', 'materials_ParquetDB')
    
    molecules_dir = os.path.join(external_dir,'materials_project','molecules')
    molecules_parquetdb_dir=os.path.join(external_dir,'materials_project', 'molecules_ParquetDB')
    
    # Using the Mprester API
    # with MPRester(api_key=MP_API_KEY) as mpr:
    #     elasticity_doc = mpr.elasticity.search(material_ids=["mp-66"])
    #     print(dir(elasticity_doc[0]))
    #     print(elasticity_doc)
    
    ################################################################################################
    ################################################################################################
    # Download stable materials
    ################################################################################################
    ################################################################################################

    # Initial screening of materials
    
    # materials_filter={
    #                 'nsites_max':40,
    #                 'energy_above_hull_min':0,
    #                 'energy_above_hull_max':0.2
    #                 }
    # # materials_filter={
    # #                 'nsites_max':2,
    # #                 'energy_above_hull_min':0,
    # #                 'energy_above_hull_max':0.025
    # #                 }
    
    
    endpoint='materials_summary'
    # download_materials(save_dir=os.path.join(materials_dir,endpoint), chunk_size=10000, **materials_filter)

    # db=ParquetDB(endpoint, dir=parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     # for entry in data['entries']:
    #     #     for site in entry['structure_graph']['structure']['sites']:
    #     #         for species in site['species']:
    #     #             species['spin']=0
    #     #             if 'spin' in species.keys():
    #     #                 species.pop('spin')
    #     db.create(data['entries'], treat_fields_as_ragged=['csm','valences'])
    # table=db.read()
    # print(table.shape)
    

    ###################################################################################################
    ###################################################################################################
    # Download materials data
    ###################################################################################################
    ###################################################################################################
    
    
    # db=ParquetDB('materials_summary', dir=materials_parquetdb_dir)
    # material_ids_table = db.read(columns=['material_id'])
    # material_ids = material_ids_table['material_id'].combine_chunks().to_pylist()
    
    ###################################################################################################
    # endpoint='chemenv'
    # # download_materials_data(materials_dir, endpoint=endpoint, material_ids=material_ids)
    # db = ParquetDB(dataset_name=endpoint, dir=materials_parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     # for entry in data['entries']:
    #     #     for site in entry['structure_graph']['structure']['sites']:
    #     #         for species in site['species']:
    #     #             species['spin']=0
    #     #             if 'spin' in species.keys():
    #     #                 species.pop('spin')
    #     db.create(data['entries'], treat_fields_as_ragged=['csm','valences'])
    # table=db.read()
    # print(table.shape)
    
    
    # #################################################################################################
    
    # endpoint='elasticity'
    # # download_materials_data(materials_dir, endpoint=endpoint, material_ids=material_ids)
    # db = ParquetDB(dataset_name=endpoint, dir=materials_parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     db.create(data['entries'], treat_fields_as_ragged=['fitting_data'])
    # table=db.read()
    # print(table.shape)

    # #################################################################################################
    
    # endpoint='bonds'
    # # download_materials_data(materials_dir, endpoint=endpoint, material_ids=material_ids)
    # db = ParquetDB(dataset_name=endpoint, dir=materials_parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     for entry in data['entries']:
    #         for site in entry['structure_graph']['structure']['sites']:
    #             for species in site['species']:
    #                 species['spin']=0
    #                 if 'spin' in species.keys():
    #                     species.pop('spin')
    #     db.create(data['entries'], treat_fields_as_ragged=['bond_length_stats','bond_types'])
    # table=db.read()
    # print(table.shape)
    
    # ################################################################################################
    
    # endpoint='piezoelectric'
    # # download_materials_data(materials_dir, endpoint=endpoint, material_ids=material_ids)
    # db = ParquetDB(dataset_name=endpoint, dir=materials_parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     # for entry in data['entries']:
    #     #     for site in entry['structure_graph']['structure']['sites']:
    #     #         for species in site['species']:
    #     #             species['spin']=0
    #     #             if 'spin' in species.keys():
    #     #                 species.pop('spin')
    #     db.create(data['entries'])
    # table=db.read()
    # print(table.shape)

    # ################################################################################################
    
    # endpoint='thermo'
    # # download_materials_data(materials_dir, endpoint=endpoint, material_ids=material_ids)
    # db = ParquetDB(dataset_name=endpoint, dir=materials_parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     for entry in data['entries'][:]:
    #         entry.pop('entries')
    #         # for key,value in entry:
    #         #     print(key,value)
    #         #     entry['entries']
                
    #     #     for site in entry['structure_graph']['structure']['sites']:
    #     #         for species in site['species']:
    #     #             species['spin']=0
    #     #             if 'spin' in species.keys():
    #     #                 species.pop('spin')
    #     db.create(data['entries'])
    # table=db.read()
    # print(table.shape)

    # ################################################################################################
    
    # endpoint='dielectric'
    # # download_materials_data(materials_dir, endpoint=endpoint, material_ids=material_ids)
    # db = ParquetDB(dataset_name=endpoint, dir=materials_parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     # for entry in data['entries'][:]:
    #         # entry.pop('entries')
    #         # for key,value in entry:
    #         #     print(key,value)
    #         #     entry['entries']
                
    #     #     for site in entry['structure_graph']['structure']['sites']:
    #     #         for species in site['species']:
    #     #             species['spin']=0
    #     #             if 'spin' in species.keys():
    #     #                 species.pop('spin')
    #     db.create(data['entries'])
    # table=db.read()
    # print(table.shape)

    # ################################################################################################
    
    # endpoint='oxidation_states'
    # # download_materials_data(materials_dir, endpoint=endpoint, material_ids=material_ids)
    # db = ParquetDB(dataset_name=endpoint, dir=materials_parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     for entry in data['entries'][:]:
    #         for site in entry['structure']['sites']:
    #             for species in site['species']:
    #                 species['spin']=0
    #                 if 'spin' in species.keys():
    #                     species.pop('spin')
    #     db.create(data['entries'])
    # table=db.read()
    # print(table.shape)
    
    
    # ################################################################################################
    
    # endpoint='electronic_structure'
    # # download_materials_data(materials_dir, endpoint=endpoint, material_ids=material_ids)
    # db = ParquetDB(dataset_name=endpoint, dir=materials_parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     # for entry in data['entries'][:]:
    #         # entry.pop('entries')
    #         # for key,value in entry:
    #         #     print(key,value)
    #         #     entry['entries']
                
    #     #     for site in entry['structure_graph']['structure']['sites']:
    #     #         for species in site['species']:
    #     #             species['spin']=0
    #     #             if 'spin' in species.keys():
    #     #                 species.pop('spin')
    #     db.create(data['entries'])
    # table=db.read()
    # print(table.shape)

    # ################################################################################################
    
    # endpoint='phonon'
    # # download_materials_data(materials_dir, endpoint=endpoint, material_ids=material_ids)
    # db = ParquetDB(dataset_name=endpoint, dir=materials_parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     # for entry in data['entries'][:]:
    #         # entry.pop('entries')
    #         # for key,value in entry:
    #         #     print(key,value)
    #         #     entry['entries']
                
    #     #     for site in entry['structure_graph']['structure']['sites']:
    #     #         for species in site['species']:
    #     #             species['spin']=0
    #     #             if 'spin' in species.keys():
    #     #                 species.pop('spin')
    #     db.create(data['entries'], convert_to_fixed_shape=False)
    # table=db.read()
    # print(table.shape)


    # ################################################################################################
    
    # endpoint='absorption'
    # # download_materials_data(materials_dir, endpoint=endpoint, material_ids=material_ids)
    # db = ParquetDB(dataset_name=endpoint, dir=materials_parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     # for entry in data['entries'][:]:
    #         # entry.pop('entries')
    #         # for key,value in entry:
    #         #     print(key,value)
    #         #     entry['entries']
                
    #     #     for site in entry['structure_graph']['structure']['sites']:
    #     #         for species in site['species']:
    #     #             species['spin']=0
    #     #             if 'spin' in species.keys():
    #     #                 species.pop('spin')
    #     db.create(data['entries'])
    # table=db.read()
    # print(table.shape)

    # ################################################################################################
    
    # endpoint='eos'
    # # download_materials_data(materials_dir, endpoint=endpoint, material_ids=material_ids)
    # db = ParquetDB(dataset_name=endpoint, dir=materials_parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     # for entry in data['entries'][:]:
    #     #     for site in entry['structure']['sites']:
    #     #         for species in site['species']:
    #     #             species['spin']=0
    #     #             if 'spin' in species.keys():
    #     #                 species.pop('spin')
    #     db.create(data['entries'], treat_fields_as_ragged=['energies','volumes'])
    # table=db.read()
    # print(table.shape)

    # ################################################################################################
    
    # endpoint='grain_boundaries'
    # download_materials_data(materials_dir, endpoint=endpoint, material_ids=material_ids)
    # db = ParquetDB(dataset_name=endpoint, dir=materials_parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     for entry in data['entries'][:1]:
    #         for key,value in entry.items():
    #             print(key,value)
    #         for site in entry['initi']['sites']:
    #             for species in site['species']:
    #                 species['spin']=0
    #                 if 'spin' in species.keys():
    #                     species.pop('spin')
    #     db.create(data['entries'],convert_to_fixed_shape=False)
    # table=db.read()
    # print(table.shape)

    # ################################################################################################

    # endpoint='provenance'
    # download_materials_data(materials_dir, endpoint=endpoint, material_ids=material_ids)
    # db = ParquetDB(dataset_name=endpoint, dir=materials_parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     # for entry in data['entries'][:]:
    #         # entry.pop('entries')
    #         # for key,value in entry:
    #         #     print(key,value)
    #         #     entry['entries']
                
    #     #     for site in entry['structure_graph']['structure']['sites']:
    #     #         for species in site['species']:
    #     #             species['spin']=0
    #     #             if 'spin' in species.keys():
    #     #                 species.pop('spin')
    #     db.create(data['entries'])
    # table=db.read()
    # print(table.shape)

    # ################################################################################################
    
    # endpoint='magnetism'
    # # download_materials_data(materials_dir, endpoint=endpoint, material_ids=material_ids)
    # db = ParquetDB(dataset_name=endpoint, dir=materials_parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     # for entry in data['entries'][:]:
    #         # entry.pop('entries')
    #         # for key,value in entry:
    #         #     print(key,value)
    #         #     entry['entries']
                
    #     #     for site in entry['structure_graph']['structure']['sites']:
    #     #         for species in site['species']:
    #     #             species['spin']=0
    #     #             if 'spin' in species.keys():
    #     #                 species.pop('spin')
    #     db.create(data['entries'], treat_fields_as_ragged=['magmoms'])
    # table=db.read()
    # print(table.shape)

    # ################################################################################################
    
    # endpoint='robocrys'
    # download_materials_data(materials_dir, endpoint=endpoint, material_ids=material_ids, use_search_docs=True)
    # db = ParquetDB(dataset_name=endpoint, dir=materials_parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     # for entry in data['entries'][:]:
    #         # entry.pop('entries')
    #         # for key,value in entry:
    #         #     print(key,value)
    #         #     entry['entries']
                
    #     #     for site in entry['structure_graph']['structure']['sites']:
    #     #         for species in site['species']:
    #     #             species['spin']=0
    #     #             if 'spin' in species.keys():
    #     #                 species.pop('spin')
    #     db.create(data['entries'])
    # table=db.read()
    # print(table.shape)

    # ################################################################################################
    
    # endpoint='similarity'
    # download_materials_data(materials_dir, endpoint=endpoint, material_ids=material_ids)
    # db = ParquetDB(dataset_name=endpoint, dir=materials_parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     # for entry in data['entries'][:]:
    #         # entry.pop('entries')
    #         # for key,value in entry:
    #         #     print(key,value)
    #         #     entry['entries']
                
    #     #     for site in entry['structure_graph']['structure']['sites']:
    #     #         for species in site['species']:
    #     #             species['spin']=0
    #     #             if 'spin' in species.keys():
    #     #                 species.pop('spin')
    #     db.create(data['entries'])
    # table=db.read()
    # print(table.shape)
    
    # ################################################################################################

    # endpoint='summary'
    # download_materials_data(materials_dir, endpoint=endpoint, material_ids=material_ids)
    # db = ParquetDB(dataset_name=endpoint, dir=materials_parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     # for entry in data['entries'][:]:
    #         # entry.pop('entries')
    #         # for key,value in entry:
    #         #     print(key,value)
    #         #     entry['entries']
                
    #     #     for site in entry['structure_graph']['structure']['sites']:
    #     #         for species in site['species']:
    #     #             species['spin']=0
    #     #             if 'spin' in species.keys():
    #     #                 species.pop('spin')
    #     db.create(data['entries'])
    # table=db.read()
    # print(table.shape)
    
    # ################################################################################################
    
    # endpoint='synthesis'
    # download_materials_data(materials_dir, endpoint=endpoint, material_ids=material_ids)
    # db = ParquetDB(dataset_name=endpoint, dir=materials_parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     # for entry in data['entries'][:]:
    #         # entry.pop('entries')
    #         # for key,value in entry:
    #         #     print(key,value)
    #         #     entry['entries']
                
    #     #     for site in entry['structure_graph']['structure']['sites']:
    #     #         for species in site['species']:
    #     #             species['spin']=0
    #     #             if 'spin' in species.keys():
    #     #                 species.pop('spin')
    #     db.create(data['entries'])
    # table=db.read()
    # print(table.shape)
    
    
    # ################################################################################################
    
    # endpoint='surface_properties'
    # download_materials_data(materials_dir, endpoint=endpoint, material_ids=material_ids)
    # db = ParquetDB(dataset_name=endpoint, dir=materials_parquetdb_dir)
    # chunk_files = glob(os.path.join(materials_dir, endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     # for entry in data['entries'][:]:
    #         # entry.pop('entries')
    #         # for key,value in entry:
    #         #     print(key,value)
    #         #     entry['entries']
                
    #     #     for site in entry['structure_graph']['structure']['sites']:
    #     #         for species in site['species']:
    #     #             species['spin']=0
    #     #             if 'spin' in species.keys():
    #     #                 species.pop('spin')
    #     db.create(data['entries'])
    # table=db.read()
    # print(table.shape)
    
    
    # ################################################################################################
    

    
    ################################################################################################
    ################################################################################################
    # Download stable molecules
    ################################################################################################
    ################################################################################################

    # Initial screening of materials
    
    endpoint='molecules_summary'
    download_molecules(save_dir=os.path.join(molecules_dir, endpoint), chunk_size=100000)
    db=ParquetDB(endpoint, dir=molecules_parquetdb_dir)
    chunk_files = glob(os.path.join(molecules_dir, endpoint, "*.json"))
    db.drop_dataset()
    for i,chunk_file in enumerate(chunk_files[:]):
        print("Processing", chunk_file)
        with open(chunk_file, "r") as f:
            data = json.load(f)
        # for entry in data['entries']:
        #     for site in entry['structure_graph']['structure']['sites']:
        #         for species in site['species']:
        #             species['spin']=0
        #             if 'spin' in species.keys():
        #                 species.pop('spin')
        db.create(data['entries'])
    table=db.read()
    print(table.shape)
    
    
    
    ###################################################################################################
    ###################################################################################################
    # Download molecules data
    ###################################################################################################
    ###################################################################################################
    
    
    # endpoint='synthesis'
    # download_materials_data(os.path.join(external_dir,'materials_project'), endpoint=endpoint, material_ids=material_ids)
    # db = ParquetDB(dataset_name=endpoint, dir=parquetdb_dir)
    # chunk_files = glob(os.path.join(external_dir, "materials_project", endpoint, "*.json"))
    # db.drop_dataset()
    # for i,chunk_file in enumerate(chunk_files[:]):
    #     print("Processing", chunk_file)
    #     with open(chunk_file, "r") as f:
    #         data = json.load(f)
    #     # for entry in data['entries'][:]:
    #         # entry.pop('entries')
    #         # for key,value in entry:
    #         #     print(key,value)
    #         #     entry['entries']
                
    #     #     for site in entry['structure_graph']['structure']['sites']:
    #     #         for species in site['species']:
    #     #             species['spin']=0
    #     #             if 'spin' in species.keys():
    #     #                 species.pop('spin')
    #     db.create(data['entries'])
    # table=db.read()
    # print(table.shape)
    
    
    
    
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
