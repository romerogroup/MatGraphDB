import os
import json
from glob import glob
from matgraphdb import config
import time
from parquetdb import ParquetDB
import pyarrow as pa
from pyarrow import compute as pc

def write_schema_summary(materials_parquetdb_dir,endpoint='chemenv', append_string=''):

    db=ParquetDB(db_path=os.path.join(materials_parquetdb_dir,endpoint))
    table=db.read()
    print(table.shape)
    
    with open(os.path.join(materials_parquetdb_dir,f'{endpoint}_schema_summary{append_string}.txt'), 'w') as f:
        f.write(f"Number of rows: {table.shape[0]}\n")
        f.write(f"Number of columns: {table.shape[1]}\n\n")
        f.write('-'*100+'\n\n')
        
        f.write(f"{'Field Name':<50} | {'Field Type'}\n")
        f.write('-'*50+'\n')
        for field in table.schema:
            f.write(f"{field.name:<50} | {field.type}\n")
            


def main():
    # Example usage
    materials_db = ParquetDB(db_path=os.path.join(config.data_dir,'materials'))
    table=materials_db.read()


    # print(table.shape)
    # data_dir=os.path.join(config.data_dir,'external','materials_project','materials','elasticity','elasticity_chunk_0.json')
    # with open(data_dir,'r') as f:
    #     data=json.load(f)
    
    # print(data['entries'][0])
    
    mp_dir=os.path.join(config.data_dir,'external','materials_project', 'materials_ParquetDB')
    mp_db=ParquetDB(db_path=os.path.join(mp_dir,'elasticity'))
    
    table=mp_db.read()



    columns_to_keep={
        'bulk_modulus.reuss': 'elasticity.k_reuss',
        'bulk_modulus.voigt': 'elasticity.k_voigt',
        'bulk_modulus.vrh': 'elasticity.k_vrh',
        'compliance_tensor.ieee_format': 'elasticity.compliance_tensor_ieee_format',
        'compliance_tensor.raw': 'elasticity.compliance_tensor_raw',
        'elastic_tensor.ieee_format': 'elasticity.elastic_tensor_ieee_format',
        'elastic_tensor.raw': 'elasticity.elastic_tensor_raw',
        'debye_temperature': 'elasticity.debye_temperature',
        'homogeneous_poisson': 'elasticity.homogeneous_poisson',
        'order': 'elasticity.order',
        'shear_modulus.reuss': 'elasticity.g_reuss',
        'shear_modulus.voigt': 'elasticity.g_voigt',
        'shear_modulus.vrh': 'elasticity.g_vrh',
        'sound_velocity.longitudinal': 'elasticity.sound_velocity_longitudinal',
        'sound_velocity.snyder_acoustic': 'elasticity.sound_velocity_acoustic',
        'sound_velocity.snyder_optical': 'elasticity.sound_velocity_optical',
        'sound_velocity.snyder_total': 'elasticity.sound_velocity_total',
        'sound_velocity.transverse': 'elasticity.sound_velocity_transverse',
        'state': 'elasticity.state',
        'thermal_conductivity.cahill': 'elasticity.thermal_conductivity_cahill',
        'thermal_conductivity.clarke': 'elasticity.thermal_conductivity_clarke',
        'universal_anisotropy': 'elasticity.universal_anisotropy',
        'young_modulus': 'elasticity.young_modulus',
        'material_id': 'core.material_id',
    }
    
    
    mp_table=mp_db.read(columns=list(columns_to_keep.keys()))
    
    # Rename columns such that they are compatible with the materials database
    mp_table = mp_table.rename_columns(columns_to_keep)
    print(mp_table.shape)

    # Append the ids of the of the materials database to the materials project database
    mp_table_with_ids=merge_materials_ids(materials_db, mp_table)
    print(mp_table_with_ids.shape)
    
    # Checking the data
    mp_df=mp_table_with_ids.to_pandas()
    print(mp_df.head())
    properties_to_check=['id','core.material_id','elasticity.k_vrh','elasticity.k_voigt','elasticity.universal_anisotropy']
    materials_table = materials_db.read(columns=properties_to_check,
                                        filters=[pc.field('id').isin([24748,15855,37359,4020,62371])])
    materials_df=materials_table.to_pandas()
    print(materials_df.head())
    
    # Compare the results between materials_df and mp_df for the selected IDs
    print("\nComparing materials database and materials project values:")
    for idx in materials_df['id']:
        mp_row = mp_df[mp_df['id'] == idx]
        materials_row = materials_df[materials_df['id'] == idx]
        
        if not mp_row.empty and not materials_row.empty:
            print(f"\nID: {idx}")
            for prop in properties_to_check:
                mp_val = mp_row[prop].iloc[0]
                mat_val = materials_row[prop].iloc[0]
                are_equal = mp_val == mat_val
                print(f"{prop}:")
                print(f"  Materials DB:        {mat_val}")
                print(f"  Materials Project:   {mp_val}")
                print(f"  Are equal:           {are_equal}")
    

    materials_db.update(mp_table_with_ids)
    
    
    write_schema_summary(config.data_dir,endpoint='materials')
    
    
        
def merge_materials_ids(materials_db, mp_table):
    """
    Merge materials database IDs with materials project table.
    
    Args:
        materials_db: ParquetDB instance for materials database
        mp_table: PyArrow table containing materials project data
        mp_materials_ids: Combined chunks of material IDs from materials project
        
    Returns:
        PyArrow table with merged IDs
    """
    # Read main table with id columen and the column to merge on ('core.material_id')
    main_table = materials_db.read(columns=['core.material_id','id'])
    print(main_table.shape)
    
    # Determine the indices of the materials database in the materials_project database
    material_indices = pc.index_in(mp_table['core.material_id'].combine_chunks(), main_table['core.material_id'].combine_chunks())
    
    # Get the ids of the materials database
    ids = pc.take(main_table['id'].combine_chunks(), material_indices)

    # Check if ids are valid
    non_null_count = pc.sum(pc.is_valid(ids))
    print(f"Number of non-null ids: {non_null_count}")
    
    # Append the ids to the materials_project database
    mp_table_with_ids=mp_table.append_column('id', ids)
    
    # Filter out rows with null ids
    mp_table_with_ids = mp_table_with_ids.filter(pc.is_valid(mp_table_with_ids['id']))
    
    return mp_table_with_ids
    
    
    
    
    
if __name__ == "__main__":
    main()
