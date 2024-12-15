import os
import json
from glob import glob
from matgraphdb import config
import time
from parquetdb import ParquetDB
import pyarrow as pa
from pyarrow import compute as pc



def main():
    # Example usage
    output_directory = os.path.join(config.data_dir,'production','materials_project','chunked_json')
    parquetdb_dir = os.path.join(config.data_dir,'production','materials_project','ParquetDB')
    dataset_name='materials'
    from_scratch=False
    
    db = ParquetDB(db_path=os.path.join(parquetdb_dir, dataset_name))
    db_test = ParquetDB(db_path=os.path.join(parquetdb_dir, dataset_name + '_test'))
    db_formated = ParquetDB(db_path=os.path.join(parquetdb_dir, dataset_name + '_formated'))
    
    table = db_formated.read()
        
    print(table.shape)
    
    for field in table.schema:
        print(field.name, field.type)
    
    print(f"Total size of loaded table: {table.nbytes / (1024 * 1024):.2f} MB")
    
    if from_scratch:
        db_test.drop_dataset()
        db_formated.drop_dataset()
    
        table = db.read()
        
        print(table.shape)
        
        for field in table.schema:
            print(field.name, field.type)
        
        print(f"Total size of loaded table: {table.nbytes / (1024 * 1024):.2f} MB")
        
        file_size_mb = os.path.getsize('/users/lllang/SCRATCH/projects/MatGraphDB/data/production/materials_project/ParquetDB/materials/materials_0.parquet') / (1024 * 1024)  # Convert bytes to MB
        
        print(f"File size: {file_size_mb:.2f} MB")
        
        ####################################################################################################################
        # Format feature vector columns
        
        table = format_feature_vectors(table)
        db_test.create(table)
        table = db_test.read()
        print(table.shape)
        for field in table.schema:
            print(field.name, field.metadata)
        file_size_mb = os.path.getsize('/users/lllang/SCRATCH/projects/MatGraphDB/data/production/materials_project/ParquetDB/materials_test/materials_test_0.parquet') / (1024 * 1024)  # Convert bytes to MB
        print(f"File size: {file_size_mb:.2f} MB")
        
        ####################################################################################################################
        # Dropping composition and oxidation state related fields
        
        drop_fields(db_test)
        table = db_test.read()
        for i, field in enumerate(table.schema):
            print(i, field.name, field.type)
        
        # ####################################################################################################################
        
        # # Rename columns
        rename_columns(db_test, db_formated)
    
    ###################################################################################################################
    
    # table = db_formated.read()
    # for field in table.schema:
    #     print(field.name, field.type)
    


def delete_null_material_ids(db):
    table = db.read()
    
    null_material_ids = pc.is_null(table['material_id'])
    null_indices = pc.indices_nonzero(null_material_ids)
    
    n_null_material_ids = len(null_indices)
    n_valid_material_ids = table.num_rows - n_null_material_ids
    print(f"Number of null material IDs: {n_null_material_ids}")
    print(f"Number of valid material IDs: {n_valid_material_ids}")
    
    null_table = table.filter(null_material_ids)
    print(null_table.num_rows)
    db.delete(ids = null_table['id'].combine_chunks().to_pylist())
    
    table = db.read()
    print(table.shape)
    null_material_ids = pc.is_null(table['material_id'])
    null_indices = pc.indices_nonzero(null_material_ids)
    
    n_null_material_ids = len(null_indices)
    n_valid_material_ids = table.num_rows - n_null_material_ids
    print(f"Number of null material IDs: {n_null_material_ids}")
    print(f"Number of valid material IDs: {n_valid_material_ids}")

def drop_fields(db):
    """Drop composition and oxidation state related fields from the database.
    
    Args:
        db: ParquetDB instance to modify
    """
    fields_to_drop = []
    for field in db.get_schema():
        if field.name.startswith('composition'):
            fields_to_drop.append(field.name)
        if field.name.startswith('oxidation_states.average_oxidation_states'):
            fields_to_drop.append(field.name)
            
    fields_to_drop.append('possible_species')
    fields_to_drop.append('bond_cutoff_connections')
    fields_to_drop.append('k_reuss')
    fields_to_drop.append('k_voigt')
    fields_to_drop.append('k_vrh')
    fields_to_drop.append('g_reuss')
    fields_to_drop.append('g_voigt')
    fields_to_drop.append('g_vrh')
    fields_to_drop.append('homogeneous_poisson')
    fields_to_drop.append('has_reconstructed')
    fields_to_drop.append('universal_anisotropy')
    db.delete(columns=fields_to_drop)
    

def add_field_metadata(table, field_name: str, metadata: dict):
    field = table.schema.field(field_name)
    
    field_metadata = field.metadata
    if field_metadata is None:
        field_metadata = {}
    field_metadata.update(metadata)
    field = field.with_metadata(field_metadata)
    field_index = table.schema.get_field_index(field_name)
    schema = table.schema.set(field_index, field)
    return table.cast(schema)


def format_feature_vectors(table):
        """
        Format feature vector columns by renaming value columns and adding metadata with feature names.
        Also drops unnecessary feature name and combined feature columns.
        """
        # Format main feature vector columns
        feature_types = [
            'element_fraction',
            'element_property', 
            'sine_coulomb_matrix',
            'xrd_pattern'
        ]
        
        for feat_type in feature_types:
            # Rename values column
            old_name = f'feature_vectors.{feat_type}.values'
            new_name = f'feature_vectors.{feat_type}'
            table = table.rename_columns({old_name: new_name})
            
            # Add feature names as metadata
            feat_names = table[f'feature_vectors.{feat_type}.feature_names'].combine_chunks()[0].as_py()
            table = add_field_metadata(table, new_name, {'feature_labels': str(feat_names)})
        
        # Drop unnecessary columns
        columns_to_drop = [
            'feature_vectors.element_fraction.feature_names',
            'feature_vectors.xrd_pattern.feature_names',
            'feature_vectors.sine_coulomb_matrix.feature_names', 
            'feature_vectors.element_property.feature_names',
            'feature_vectors.sine_coulomb_matrix-element_property.feature_names',
            'feature_vectors.sine_coulomb_matrix-element_property.values',
            'feature_vectors.sine_coulomb_matrix-element_property-element_fraction.feature_names',
            'feature_vectors.sine_coulomb_matrix-element_property-element_fraction.values',
            'feature_vectors.sine_coulomb_matrix-element_fraction.feature_names',
            'feature_vectors.sine_coulomb_matrix-element_fraction.values',
            'feature_vectors.element_property-element_fraction.feature_names',
            'feature_vectors.element_property-element_fraction.values',
            
            'id'
        ]
        
        table = table.drop(columns=columns_to_drop)
        return table
    
    
def rename_columns(db, new_db):
    table = db.read()
    
    field_to_rename={
        'total_magnetization': 'magnetism.total_magnetization',
        'total_magnetization_normalized_vol': 'magnetism.total_magnetization_normalized_vol', 
        'types_of_magnetic_species': 'magnetism.types_of_magnetic_species',
        'num_magnetic_sites': 'magnetism.num_magnetic_sites',
        'num_unique_magnetic_sites': 'magnetism.num_unique_magnetic_sites',
        'ordering': 'magnetism.ordering',
        
        'chargemol_bonding_connections': 'chargemol.bond_connections',
        'chargemol_bonding_orders': 'chargemol.bond_orders',
        'chargemol_cubed_moments': 'chargemol.cubed_moments',
        'chargemol_fourth_moments': 'chargemol.fourth_moments',
        'chargemol_squared_moments': 'chargemol.squared_moments',
        
        'coordination_environments_multi_weight': 'chemenv.coordination_environments_multi_weight',
        'coordination_multi_connections': 'chemenv.coordination_multi_connections',
        'coordination_multi_numbers': 'chemenv.coordination_multi_numbers',
        
        'electric_consistent_bond_connections': 'bonding.electric_consistent.bond_connections',
        'electric_consistent_bond_orders': 'bonding.electric_consistent.bond_orders',
        
        'geometric_consistent_bond_connections': 'bonding.geometric_consistent.bond_connections',
        'geometric_consistent_bond_orders': 'bonding.geometric_consistent.bond_orders',
        
        'geometric_electric_consistent_bond_connections': 'bonding.geometric_electric_consistent.bond_connections',
        'geometric_electric_consistent_bond_orders': 'bonding.geometric_electric_consistent.bond_orders',
        
        'bonding_cutoff_connections': 'bonding.cutoff_method.bond_connections',
        
        'n': 'dielectric.n',
        'e_ionic': 'dielectric.e_ionic',
        'e_electronic': 'dielectric.e_electronic',
        'e_total': 'dielectric.e_total',
        'e_ij_max': 'dielectric.e_ij_max',
        
        'decomposes_to': 'thermo.decomposes_to',
        'formation_energy_per_atom': 'thermo.formation_energy_per_atom',
        'energy_above_hull': 'thermo.energy_above_hull',
        'uncorrected_energy_per_atom': 'thermo.uncorrected_energy_per_atom',
        'equilibrium_reaction_energy_per_atom': 'thermo.equilibrium_reaction_energy_per_atom',
        
        'formula_pretty': 'core.formula_pretty',
        'nelements': 'core.nelements',
        'nsites': 'core.nsites',
        'volume': 'core.volume',
        'density_atomic': 'core.density_atomic',
        'density': 'core.density',
        'energy_per_atom': 'core.energy_per_atom',
        'elements': 'core.elements',
        'is_gap_direct': 'core.is_gap_direct',
        'is_magnetic': 'core.is_magnetic',
        'is_metal': 'core.is_metal',
        'material_id': 'core.material_id',
        'is_stable': 'core.is_stable',
        
        'vbm': 'electronic_structure.vbm',
        'cbm': 'electronic_structure.cbm',
        'band_gap': 'electronic_structure.band_gap',
        'dos_energy_up': 'electronic_structure.dos_energy_up',
        'efermi': 'electronic_structure.efermi',
        
        'last_updated': 'metadata.last_updated',
        'theoretical': 'metadata.theoretical',
        
        'weighted_surface_energy': 'surface_properties.weighted_surface_energy',
        'weighted_surface_energy_EV_PER_ANG2': 'surface_properties.weighted_surface_energy_EV_PER_ANG2',
        'weighted_work_function': 'surface_properties.weighted_work_function',
        'surface_anisotropy': 'surface_properties.surface_anisotropy',
        'shape_factor' : 'surface_properties.shape_factor',
        
        'wyckoffs': 'symmetry.wyckoffs',
        
        'grain_boundaries': 'grain_boundaries.grain_boundaries',
    }

    table = table.rename_columns(field_to_rename)
    
    
    table = table.drop(columns=['id'])
    new_db.create(table) 
    
    
    table = new_db.read()
    schema= table.schema
    new_db.update_schema(schema=schema)
    
    # schema = db.get_schema()
    
    # new_fields=[]
    # for field in schema:
    #     if field.name in field_to_rename:
    #         new_field_name= field_to_rename[field.name]
    #         new_fields.append(pa.field(new_field_name, field.type))
    # new_schema = pa.schema(new_fields)
    # db.update_schema(schema=new_schema)
    
    


    
    
    
    
    
    
   





if __name__ == "__main__":
    
    main()
    