import os
import shutil
import json


from poly_graphs_lib.utils import PROJECT_DIR
from poly_graphs_lib.cfg import apikey
from mp_api.client import MPRester

print(apikey)

# with MPRester(apikey) as mpr:
#     summary_docs = mpr.summary._search( nelements= 1, energy_above_hull_min = 0, energy_above_hull_max = 0.02, fields=['material_id'])
# summary_doc_dict = summary_docs[0].dict()
# print('------------------------------------------------')
# print('Possible fields')
# print('------------------------------------------------')
# for key in list(summary_doc_dict.keys())[1:]:
#     for field in summary_doc_dict[key]:
#         print(field)
# print('------------------------------------------------')

fileds_to_include=['material_id','nsites','elements','nelements','composition',
                   'composition_reduced','formula_pretty','volume',
                   'density','density_atomic','symmetry','structure',
                   'energy_per_atom','formation_energy_per_atom','energy_above_hull','is_stable',
                   'band_gap','cbm','vbm','is_stable','efermi','is_gap_direct','is_metal',
                   'is_magnetic','ordering','total_magnetization','total_magnetization_normalized_vol',
                   'num_magnetic_sites','num_unique_magnetic_sites',
                   'k_voigt','k_reuss','k_vrh','g_voigt','g_reuss','g_vrh',
                   'universal_anisotropy','homogeneous_poisson','e_total','e_ionic','e_electronic']


with MPRester(apikey) as mpr:
    summary_docs = mpr.summary._search( nelements=2, energy_above_hull_min=0, energy_above_hull_max=0.01, fields=fileds_to_include)

# print('------------------------------------------------')
# print("Generating single json file database")
# print('------------------------------------------------')
# json_database={}
# print('------------------------------------------------')
# for doc in summary_docs:
#     summary_doc_dict = doc.dict()

#     mp_id=summary_doc_dict['material_id']
#     json_database.update({mp_id:{}})
#     for field_name in fileds_to_include:
#         json_database[mp_id].update({field_name:summary_doc_dict[field_name]})

# json_file=os.path.join(PROJECT_DIR,'data','raw','json_database_7235.json')
# with open(json_file, 'w') as f:
#     json.dump(json_database, f, indent=4)
# print('------------------------------------------------')



print('------------------------------------------------')
print("Generating directory json files database")
print('------------------------------------------------')
print('------------------------------------------------')
json_database_dir=os.path.join(PROJECT_DIR,'data','raw','mp_database')
shutil.rmtree(json_database_dir)
for doc in summary_docs:
    summary_doc_dict = doc.dict()
    mp_id=summary_doc_dict['material_id']

    json_file=os.path.join(PROJECT_DIR,'data','raw','mp_database',f'{mp_id}.json')

    json_database_entry={}
    for field_name in fileds_to_include:
        json_database_entry.update({field_name:summary_doc_dict[field_name]})

    if not os.path.exists(os.path.dirname(json_file)):
        os.makedirs(os.path.dirname(json_file))
        
    with open(json_file, 'w') as f:
        json.dump(json_database_entry, f, indent=4)
print('------------------------------------------------')

















# with MPRester(apikey) as mpr:
#     criteria={'nelements':4,
#               'energy_above_hull_min':0,
#               'energy_above_hull_max':0.02}
#     # Initial screening 
#     summary_docs = mpr.summary._search( nelements= 3, energy_above_hull_min = 0, energy_above_hull_max = 0.02, fields=['material_id'])

#     # criteria={"nelements":1 ,"e_above_hull":{"$lte":0.02} }, 
#     # summary_docs = mpr.summary._search( **criteria, fields=['material_id','structure'])
#     # print(summary_docs)