import os
import json
from glob import glob
from matgraphdb import config
import time
from parquetdb import ParquetDB
import pyarrow as pa
from pyarrow import compute as pc

from matgraphdb.utils.parquet_tools import write_schema_summary

def main():

    
    merge_elasticity_materials_db()
    
    # merge_materials_summary_with_materials_db()


def merge_materials_summary_with_materials_db():
    
    materials_db = ParquetDB(db_path=os.path.join(config.data_dir,'materials'))
    
    
    mp_dir=os.path.join(config.data_dir,'external','materials_project', 'materials_ParquetDB')
    mp_db=ParquetDB(db_path=os.path.join(mp_dir,'materials_summary'))
    
    
    materials_table=mp_db.read()
    # print(materials_table.shape)
    # for field in materials_table.schema:
    #     print(field.name, field.type)
    
    columns_to_keep={
        'band_gap': 'electronic_structure.band_gap',
        'cbm': 'electronic_structure.cbm',
        'efermi': 'electronic_structure.efermi',
        'vbm': 'electronic_structure.vbm',
        'efermi': 'electronic_structure.efermi',
        
        'symmetry.crystal_system': 'symmetry.crystal_system',
        'symmetry.number': 'symmetry.number',
        'symmetry.point_group': 'symmetry.point_group',
        'symmetry.symbol': 'symmetry.symbol',
        'symmetry.symprec': 'symmetry.symprec',
        'symmetry.version': 'symmetry.version',
        'structure.@class': 'structure.@class',
        'structure.@module': 'structure.@module',
        'structure.charge': 'structure.charge',
        'structure.lattice.a': 'structure.lattice.a',
        'structure.lattice.alpha': 'structure.lattice.alpha',
        'structure.lattice.b': 'structure.lattice.b',
        'structure.lattice.beta': 'structure.lattice.beta',
        'structure.lattice.c': 'structure.lattice.c',
        'structure.lattice.gamma': 'structure.lattice.gamma',
        'structure.lattice.matrix': 'structure.lattice.matrix',
        'structure.lattice.pbc': 'structure.lattice.pbc',
        'structure.lattice.volume': 'structure.lattice.volume',
        'structure.sites': 'structure.sites',
        
        'surface_anisotropy': 'surface_properties.surface_anisotropy',
        'weighted_surface_energy': 'surface_properties.weighted_surface_energy',
        'weighted_surface_energy_EV_PER_ANG2': 'surface_properties.weighted_surface_energy_EV_PER_ANG2',
        'weighted_work_function': 'surface_properties.weighted_work_function',
        'shape_factor': 'surface_properties.shape_factor',
        
        'xas': 'xas',
        
        'theoretical': 'metadata.theoretical',
        'last_updated': 'metadata.last_updated',
        'deprecated': 'metadata.deprecated',
        'database_IDs.icsd': 'metadata.database_IDs.icsd',
        'database_IDs.pf': 'metadata.database_IDs.pf',
        'builder_meta.build_date': 'metadata.builder_meta.build_date',
        'builder_meta.database_version': 'metadata.builder_meta.database_version',
        'builder_meta.emmet_version': 'metadata.builder_meta.emmet_version',
        'builder_meta.license': 'metadata.builder_meta.license',
        'builder_meta.pymatgen_version': 'metadata.builder_meta.pymatgen_version',
        'builder_meta.run_id': 'metadata.builder_meta.run_id',
        
        'total_magnetization': 'magnetism.total_magnetization',
        'total_magnetization_normalized_formula_units': 'magnetism.total_magnetization_normalized_formula_units',
        'total_magnetization_normalized_vol': 'magnetism.total_magnetization_normalized_vol',
        'types_of_magnetic_species': 'magnetism.types_of_magnetic_species',
        'ordering': 'magnetism.ordering',
        'num_magnetic_sites': 'magnetism.num_magnetic_sites',
        'num_unique_magnetic_sites': 'magnetism.num_unique_magnetic_sites',
        
        'shear_modulus.reuss': 'elasticity.g_reuss',
        'shear_modulus.voigt': 'elasticity.g_voigt',
        'shear_modulus.vrh': 'elasticity.g_vrh',
        'universal_anisotropy': 'elasticity.universal_anisotropy',
        'homogeneous_poisson': 'elasticity.homogeneous_poisson',
        'bulk_modulus.reuss': 'elasticity.k_reuss',
        'bulk_modulus.voigt': 'elasticity.k_voigt',
        'bulk_modulus.vrh': 'elasticity.k_vrh',
        
        'origins': 'origins',
        
        'material_id': 'core.material_id',
        'nelements': 'core.nelements',
        'nsites': 'core.nsites',
        'is_gap_direct': 'core.is_gap_direct',
        'is_magnetic': 'core.is_magnetic',
        'is_metal': 'core.is_metal',
        'is_stable': 'core.is_stable',
        'formula_anonymous': 'core.formula_anonymous',
        'formula_pretty': 'core.formula_pretty',
        'elements': 'core.elements',
        'density': 'core.density',
        'density_atomic': 'core.density_atomic',
        'chemsys': 'core.chemsys',
        
        'e_electronic': 'dielectric.e_electronic',
        'e_ij_max': 'dielectric.e_ij_max',
        'e_ionic': 'dielectric.e_ionic',
        'e_total': 'dielectric.e_total',
        
        'grain_boundaries': 'grain_boundaries.grain_boundaries',
        
        'formation_energy_per_atom': 'thermo.formation_energy_per_atom',
        'energy_above_hull': 'thermo.energy_above_hull',
        'energy_per_atom': 'thermo.energy_per_atom',
        'equilibrium_reaction_energy_per_atom': 'thermo.equilibrium_reaction_energy_per_atom',
        'decomposes_to': 'thermo.decomposes_to',
    }

    mp_table=mp_db.read(columns=list(columns_to_keep.keys()))
    
    
    
    
    materials_ids=materials_table['material_id'].combine_chunks()
    mp_ids=mp_table['material_id'].combine_chunks()
    
    
    # # Determine the indices of the materials database in the materials_project database
    # material_indices = pc.index_in(mp_table['core.material_id'].combine_chunks(), main_table['core.material_id'].combine_chunks())
    
    # # Get the ids of the materials database
    # ids = pc.take(main_table['id'].combine_chunks(), material_indices)
    
    
    # # Rename columns such that they are compatible with the materials database
    # mp_table = mp_table.rename_columns(columns_to_keep)
    # print(mp_table.shape)
    
    #  # Read main table with id columen and the column to merge on ('core.material_id')
    # main_table = materials_db.read(columns=['core.material_id','id'])
    # print(main_table.shape)
    
    # Determine the indices of the materials database in the materials_project database
    # material_indices = pc.index_in(mp_table['core.material_id'].combine_chunks(), main_table['core.material_id'].combine_chunks())


def merge_elasticity_materials_db():
    # Example usage
    materials_db = ParquetDB(db_path=os.path.join(config.data_dir,'materials'))
    table=materials_db.read()

    
    table = materials_db.read(filters=[pc.field('core.material_id') == 'mp-985554'])
    df=table.combine_chunks().to_pandas()
    print(df)
    print(df['structure.lattice.matrix'])
    

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
    

    filtered_table = mp_table_with_ids.filter(pc.field('core.material_id') == 'mp-985554')
    print(filtered_table.shape)
    
    # for x in 
    df=filtered_table.combine_chunks().to_pandas()
    
    print(filtered_table.shape)
    materials_db.update(mp_table_with_ids)
    
    
    
    table = materials_db.read(filters=[pc.field('core.material_id') == 'mp-985554'])
    df=table.combine_chunks().to_pandas()
    print(df)
    print(df['structure.lattice.matrix'])
    
    
    write_schema_summary(os.path.join(config.data_dir,'materials'), output_path=os.path.join(config.data_dir,'materials_schema','schema_summary.txt'))
    
    
        
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
