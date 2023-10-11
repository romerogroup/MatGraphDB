import os

import json
from glob import glob

from multiprocessing import Pool
from poly_graphs_lib.utils import PROJECT_DIR
from poly_graphs_lib.cfg import apikey
from mp_api.client import MPRester


# for file in database_files[:2]:
def process_entry(file):

    calcs_database_dir=os.path.join(PROJECT_DIR,'data','raw','mp_database_calcs')
    material_id=file.split(os.sep)[-1].split('.')[0]
    calcs_dir=os.path.join(calcs_database_dir,material_id,'static')
    os.makedirs(calcs_dir,exist_ok=True)
    try:
        with MPRester(apikey) as mpr:
            task_id=mpr.get_task_ids_associated_with_material_id(material_id=material_id, calc_types=['GGA Static'])[0]
            tasks_doc = mpr.tasks._search( task_ids=task_id)[0]

            orig_inputs=tasks_doc.orig_inputs
            kpoints=orig_inputs.kpoints
            poscar=orig_inputs.poscar
            potcar=orig_inputs.potcar
            incar=orig_inputs.incar

            with open(os.path.join(calcs_dir,"INCAR"),'w') as f:
                for key,value in incar.items():
                    f.write(f"{key} = {value}\n")

            with open(os.path.join(calcs_dir,"KPOINTS"),'w') as f:
                f.write(str(kpoints))

            with open(os.path.join(calcs_dir,"POSCAR"),'w') as f:
                f.write(str(poscar))

            with open(os.path.join(calcs_dir,"POTCAR_files.json"),'w') as f:

                potcar_dict={"functional":potcar.functional,
                            "symbols":potcar.symbols
                            }

                json.dump(potcar_dict,f)
    except:
        print(material_id)
        pass


def process_database(n_cores=1):
    # json_database_dir=os.path.join(PROJECT_DIR,'data','raw','mp_database')
    # database_files=glob(json_database_dir + '\*.json')
    
    database_dir=os.path.join(PROJECT_DIR,'data','raw','mp_database')
    database_files=glob(database_dir + '\*.json')
    if n_cores==1:
        for i,file in enumerate(database_files[:50]):
            if i%100==0:
                print(i)
            print(file)
            process_entry(file)
    else:
        with Pool(n_cores) as p:
            p.map(process_entry, database_files)


if __name__=='__main__':
    process_database(n_cores=6)



# with MPRester(apikey) as mpr:
#     # tasks_doc = mpr.tasks.get_data_by_id(
#     #         "mp-1000",           # task_id of this calculation
#     #         fields=["task_id", "orig_inputs", "calcs_reversed", "output", "last_updated"]
#     #     )

#     print(dir(mpr))
#     # print(mpr.get_task_ids_associated_with_material_id(material_id='mp-1000'))#, calc_types=[CalcType.GGA_STATIC]))
#     # print(mpr.materials.get_data_by_id('mp-1000'))#, fields=["calc_types"]))

#     print(mpr.get_task_ids_associated_with_material_id(material_id='mp-1000', calc_types=['GGA Static']))
#     tasks_doc = mpr.tasks._search( task_ids='mp-654763')[0]

#     # print(dir(tasks_doc))
#     # print(dir(tasks_doc.orig_inputs))
#     orig_inputs=tasks_doc.orig_inputs
    
#     kpoints=orig_inputs.kpoints
#     poscar=orig_inputs.poscar
#     potcar=orig_inputs.potcar
#     incar=orig_inputs.incar
#     potcar_ok=orig_inputs.potcar_ok

#     # print(kpoints)
#     # print(poscar)
#     print(potcar.functional)
#     print(potcar.symbols)
#     # print(incar)
