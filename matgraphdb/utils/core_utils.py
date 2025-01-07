import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

t_string=pa.string()
t_int=pa.int64()
t_float=pa.float64()
t_bool=pa.bool_()

material_property_schema_list=[
    pa.field('material_id', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('nsites', t_int, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elements', pa.list_(t_string), metadata={'encoder':'ElementsEncoder(dtype=torch.float32)'}),
    pa.field('nelements', t_int, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('formula_pretty', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('volume', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('density', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('density_atomic', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('crystal_system', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('space_group', t_int, metadata={'encoder':'SpaceGroupOneHotEncoder()'}),
    pa.field('space_group_symbol', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('point_group', t_string, metadata={'encoder':'ClassificationEncoder()'}),

    pa.field('composition-values', pa.list_(t_float)),
    pa.field('composition-elements', pa.list_(t_string)),
    pa.field('composition_reduced-values', pa.list_(t_float)),
    pa.field('composition_reduced-elements', pa.list_(t_string)),

    pa.field('lattice',pa.list_(pa.list_(t_float))),
    # pa.field('pbc', pa.list_(t_bool)),
    pa.field('a', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('b', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('c', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('alpha', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('beta', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('gamma', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('frac_coords', pa.list_(pa.list_(t_float))),
    pa.field('cart_coords', pa.list_(pa.list_(t_float))),
    pa.field('species', pa.list_(t_string)),
    pa.field('unit_cell_volume', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),

    pa.field('energy_per_atom', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('formation_energy_per_atom', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('energy_above_hull', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('band_gap', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('cbm', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('vbm', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('efermi', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('is_stable', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('is_gap_direct', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('is_metal', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('is_magnetic', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('ordering', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('total_magnetization', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('total_magnetization_normalized_vol', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('num_magnetic_sites', t_int, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('num_unique_magnetic_sites', t_int, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('k_voigt', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('k_reuss', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('k_vrh', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('g_voigt', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('g_reuss', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('g_vrh', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('universal_anisotropy', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('homogeneous_poisson', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('e_total', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('e_ionic', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('e_electronic', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),

    pa.field('sine_coulomb_matrix', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('element_fraction', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('element_property', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('xrd_pattern', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),

    pa.field('coordination_environments_multi_weight', pa.list_(pa.list_(pa.struct([('ce_symbol', t_string),
                                                                    ('ce_fraction', t_float),
                                                                    ('csm', t_float),
                                                                    ('permutation', pa.list_(t_int))])))),
    pa.field('coordination_multi_connections', pa.list_(pa.list_(t_int))),
    pa.field('coordination_multi_numbers', pa.list_(t_int), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('chargemol_bonding_connections', pa.list_(pa.list_(t_int))),
    pa.field('chargemol_bonding_orders', pa.list_(pa.list_(t_float)), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    # pa.field('chargemol_net_atomic_charges', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('chargemol_squared_moments', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('chargemol_cubed_moments', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('chargemol_fourth_moments', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('geometric_consistent_bond_connections', pa.list_(pa.list_(t_int))),
    pa.field('geometric_consistent_bond_orders', pa.list_(pa.list_(t_float)), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('electric_consistent_bond_connections', pa.list_(pa.list_(t_int))),
    pa.field('electric_consistent_bond_orders', pa.list_(pa.list_(t_float)), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('geometric_electric_consistent_bond_connections', pa.list_(pa.list_(t_int))),
    pa.field('geometric_electric_consistent_bond_orders', pa.list_(pa.list_(t_float)), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('bond_cutoff_connections', pa.list_(pa.list_(t_int))),
    pa.field('wyckoffs', pa.list_(t_string)),

    pa.field('oxidation_states-possible_species', pa.list_(t_string)),
    pa.field('oxidation_states-possible_valences', pa.list_(t_float), metadata={'encoder':'ListIdentityEncoder(dtype=torch.float32)'}),
    pa.field('oxidation_states-method', t_string, metadata={'encoder':'ClassificationEncoder()'}),

    pa.field('last_updated', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('uncorrected_energy_per_atom', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('equilibrium_reaction_energy_per_atom', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('decomposes_to', pa.list_(pa.struct([
                                ('amount', t_float),
                                ('formula', t_string),
                                ('material_id', t_string)]))),
    pa.field('types_of_magnetic_species', pa.list_(t_string)),
    pa.field('grain_boundaries', pa.list_(pa.struct([
                                ('gb_energy', t_float),
                                ('rotation_angle', t_float),
                                ('sigma', t_int),
                                ('type', t_string)]))),
    pa.field('dos_energy_up', pa.list_(t_float), metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),

    pa.field('n', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('e_ij_max', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('weighted_surface_energy_EV_PER_ANG2', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('weighted_surface_energy', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('weighted_work_function', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('surface_anisotropy', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('shape_factor', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('has_reconstructed', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('possible_species', pa.list_(t_string)),
    pa.field('theoretical', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-materials', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-thermo', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-xas', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-grain_boundaries', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-chemenv', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-electronic_structure', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-absorption', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-bandstructure', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-dos', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-magnetism', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-elasticity', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-dielectric', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-piezoelectric', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-surface_properties', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-oxi_states', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-provenance', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-charge_density', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-eos', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-phonon', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-insertion_electrodes', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('has_props-substrates', t_bool, metadata={'encoder':'BooleanEncoder(dtype=torch.int64)'}),
    pa.field('elasticity-warnings', pa.list_(t_string)),
    pa.field('elasticity-order', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-k_vrh', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-k_reuss', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-k_voigt', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-g_vrh', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-g_reuss', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-g_voigt', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-sound_velocity_transverse', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-sound_velocity_longitudinal', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-sound_velocity_total', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-sound_velocity_acoustic', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-sound_velocity_optical', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-thermal_conductivity_clarke', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-thermal_conductivity_cahill', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-young_modulus', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-universal_anisotropy', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-homogeneous_poisson', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-debye_temperature', t_float, metadata={'encoder':'IdentityEncoder(dtype=torch.float32)'}),
    pa.field('elasticity-state', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('name', t_string, metadata={'encoder':'ClassificationEncoder()'}),
    pa.field('type', t_string, metadata={'encoder':'ClassificationEncoder()'})]


MATERIAL_PARQUET_SCHEMA = pa.schema(material_property_schema_list)



# # empty_table = pa.Table.from_pandas(pd.DataFrame(columns=[field.name for field in MATERIAL_PARQUET_SCHEMA]), schema=MATERIAL_PARQUET_SCHEMA)

# output_parquet_file = 'sandbox/schema.parquet'
# # pq.write_table(empty_table, output_parquet_file)

# table = pq.read_table(output_parquet_file)

# # Extract the current schema
# current_schema = table.schema
# print("Current Schema:", current_schema)


# # Define additional fields (example)
# new_fields = [
#     pa.field('new_field_1', pa.int64(), metadata={'encoder':'ClassificationEncoder()'}), 
#     pa.field('new_field_2', pa.string(), metadata={'encoder':'ClassificationEncoder()'})
# ]

# # Create the updated schema by combining the old schema with new fields
# new_schema = pa.schema(list(current_schema) + new_fields)
# print("Updated Schema:", new_schema)

# # Convert table to pandas DataFrame to add new columns (if needed)
# df = table.to_pandas()
# df['new_field_1'] = None
# df['new_field_2'] = None

# # Create a new pyarrow Table with the updated schema and data
# new_table = pa.Table.from_pandas(df, schema=new_schema)

# # Save the updated table to a new parquet file
# updated_parquet_file = 'sandbox/updated_file.parquet'
# pq.write_table(new_table, updated_parquet_file)


# table = pq.read_table(updated_parquet_file)

# # Extract the current schema
# current_schema = table.schema
# print("Current Schema:", current_schema)
